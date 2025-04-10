"""
Train a noised image classifier on ImageNet.
"""
import wandb
import argparse
import os
import sys
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import BRATSDataset
# from guided_diffusion.litsloader import LiTSDataset

import blobfile as bf
import torch as th
# from guided_diffusion.losses import FocalLoss
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
# from visdom import Visdom
import numpy as np
# viz = Visdom(port=8850)
# loss_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='classification loss'))
# val_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='validation loss'))
# acc_window= viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='acc', title='accuracy'))
from torchvision import transforms
from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.train_util import visualize
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict


def main():

    ## 
    '''
    classifier_scale = 100
    def cond_fn(x_0, classifier, t, y=None):
        assert y is not None
        with th.enable_grad():
            # Giữ nguyên x_0 mà không detach
            logits = classifier(x_0, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]  # range(len(logits)) = batch_size
            # Tính toán gradient
            a = th.autograd.grad(selected.sum(), x_0)[0]
            return a, a * classifier_scale
    '''

    def min_max_scaler(x):
        x_flat = x.reshape(x.shape[0], -1)
        x_min = th.min(x_flat, dim=1).values
        x_max = th.max(x_flat, dim=1).values
        scale = x_max - x_min
        x_normalize = (x - x_min[:, None, None]) / scale[:, None, None]
        return x_normalize
    '''
    def saliency_map(x_0,classifier):
        t_0 = th.randint(low=0, high=1, size=(1,), device=dist_util.dev())
        ds_label = th.randint(low=0, high=1, size=(1,), device=dist_util.dev())
        x0_grad, _ = cond_fn(x_0,classifier,t_0,ds_label)
        grad_img = th.abs(th.sum(x0_grad, dim=1)) ## from (B,C,H,W) to (B,H,W) because we calculate the sum of 4 dimensions
        coarse_mask = min_max_scaler(grad_img) ## mask  
        # không dùng được vì ko truyền ngược: gaussian_blur = GaussianBlur(15, 5)
        soft_mask = th.sigmoid((0.4 - coarse_mask) * 1000) #ngưỡng 0.4 - 1/(1+e^-t)
        return soft_mask
    '''

    ###
    args = create_argparser().parse_args()

    ########
    wandb.login(key="18867541319386f8b2e1362741174bd50968c3f3")
    wandb.init(
        project="brats-regularization-classifier",  # Replace with your project name
        config=args,  # Optionally log hyperparameters
    )

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys()),
    )
    model.to(dist_util.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion, maxt=args.max_L
        )
    
    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )

    data_transform = transforms.RandomApply([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(90)
    ], p=0.5)
    transform = data_transform if args.transform else None
    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )
        
    logger.log("creating data loader...")

    if args.dataset == 'brats':
        print("Training on BRATS-20 dataset")
        ds = BRATSDataset(mode="train", fold=args.fold, test_flag=False, transforms=transform)
        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True)
        data = iter(datal)

    # elif args.dataset == 'lits':
    #     print("Training on LiTS dataset")

    #     ds = LiTSDataset(args.data_dir, mode="train", test_flag=False)
    #     datal = th.utils.data.DataLoader(
    #         ds,
    #         batch_size=args.batch_size,
    #         shuffle=True)
    #     data = iter(datal)

    try:
        val_ds = BRATSDataset(mode="test", fold=args.fold, test_flag=False)
        val_datal = th.utils.data.DataLoader(
            val_ds,
            batch_size= args.batch_size,
            shuffle=True)
        val_data = iter(val_datal)
    except:
        val_data = None

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        if os.path.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            opt.load_state_dict(
                dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
            )

    logger.log("training classifier model...")

    def validation_log(val_data_load):
        data_loader = iter(val_data_load)
        accuracies = []
        losses = []
        data_size = 0
        for data in data_loader:
            batch, _, labels, _ = data
            data_size += batch.shape[0]
            batch = batch.to(dist_util.dev())
            labels= labels.to(dist_util.dev())
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())
            for i, (sub_batch, sub_labels, sub_t) in enumerate(
                split_microbatches(args.microbatch, batch, labels, t)
            ):
            
                logits = model(sub_batch, timesteps=sub_t)
            
                loss = F.cross_entropy(logits, sub_labels, reduction="none") 
                loss = loss.mean()

                accuracy = compute_top_k(
                    logits, sub_labels, k=1, reduction="none"
                )
                losses.append(loss.mean().item())
                accuracies.append(accuracy.mean().item())
        print(f"Validation dataset size: {data_size}")

        return np.mean(losses), np.mean(accuracies)
    
    # ---------------------------------------------
    lambda_0 = 0.1 
    lambda_1 = 0.01
    lambda_2 = 0.1

    def forward_backward_log(data_load, data_loader, prefix="train"):
        try:
            batch, _, labels, masks, exist_annotations, cluster_labels = next(data_loader)
        except:
            data_loader = iter(data_load)
            batch, _, labels, masks, exist_annotations, cluster_labels = next(data_loader)

        # Di chuyển data sang device tương ứng
        batch = batch.to(dist_util.dev())
        labels = labels.to(dist_util.dev())
        masks = masks.to(dist_util.dev())
        # Giả sử mask ban đầu có shape (B, 256, 256) -> replicate theo channel để có shape (B, 4, 256, 256)
        masks = masks.unsqueeze(1).repeat(1, 4, 1, 1)
        exist_annotations = exist_annotations.to(dist_util.dev())
        cluster_labels = cluster_labels.to(dist_util.dev())

        batch_0 = batch
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())
        
        # Giả sử lớp cần chọn là 0 (với high=1, tức luôn 0)
        classes = th.randint(low=0, high=1, size=(batch_0.shape[0],), device=dist_util.dev())
        t_0 = th.zeros(batch_0.shape[0], dtype=th.long, device=dist_util.dev())

        for i, (sub_batch_0, sub_batch, sub_masks, sub_labels, sub_t, sub_t_0, sub_classes, sub_anno, sub_cluster) in enumerate(
            split_microbatches(
                args.microbatch, batch_0, batch, masks, labels, t, t_0, classes, exist_annotations, cluster_labels
            )
        ):
            # Loss phân loại từ logits
            logits = model(sub_batch, timesteps=sub_t)
            loss_cls = F.cross_entropy(logits, sub_labels, reduction="none")

            with th.enable_grad():
                # Đảm bảo sub_batch_0 tách riêng và có requires_grad=True để tính gradient
                sub_batch_0 = sub_batch_0.detach().requires_grad_(True)
                logits_0 = model(sub_batch_0, sub_t_0)
                log_probs = F.log_softmax(logits_0, dim=-1)
                selected = log_probs[range(len(logits_0)), sub_classes.view(-1)]
                # selected.sum() để tạo scalar loss; mỗi input chỉ phụ thuộc vào output của nó
                a = th.autograd.grad(selected.sum(), sub_batch_0, create_graph=True)[0]

                # ---------------------------
                # 1. Tính real_annotation_loss:
                # Với mỗi sample có exist_annotation == 1, tính norm giữa gradient và mask tương ứng,
                # còn với sample không có annotation thì đặt loss = 0.
                anno_mask = (sub_anno == 1)
                if anno_mask.sum() > 0:
                    diff = a[anno_mask] - sub_masks[anno_mask]
                    real_annotation_loss = th.norm(diff, p=2, dim=(1, 2, 3))
                else:
                    real_annotation_loss = th.tensor(0.0, device=a.device, dtype=a.dtype, requires_grad=True)

                # ---------------------------
                # 2. Tính loss_discrepancy: so sánh giữa mẫu thực (annotation==1) và mẫu sinh ra (annotation==0) trong cùng 1 cluster
                loss_discrepancy = th.tensor(0.0, device=a.device, dtype=a.dtype, requires_grad=True)
                # 3. Tính loss_robustness: đo khoảng cách giữa các mẫu sinh ra trong cùng cluster
                loss_robustness = th.tensor(0.0, device=a.device, dtype=a.dtype, requires_grad=True)

                # Vector hóa theo cluster (giả sử có 4 cluster: 0,1,2,3)
                for cluster in range(4):
                    cluster_mask = (sub_cluster == cluster)
                    # Mẫu annotated (real)
                    real_mask = anno_mask & cluster_mask
                    # Mẫu generated (không có annotation)
                    gen_mask = (~anno_mask) & cluster_mask

                    if real_mask.sum() > 0 and gen_mask.sum() > 0:
                        real_vals = a[real_mask]   # shape: (N_real, C, H, W)
                        gen_vals = a[gen_mask]     # shape: (N_gen, C, H, W)
                        # Tính cặp hiệu giữa real và generated: broadcasting
                        diffs = real_vals.unsqueeze(1) - gen_vals.unsqueeze(0)  # shape: (N_real, N_gen, C, H, W)
                        # Tính norm L2 cho mỗi cặp, flatten các chiều (C,H,W)
                        norms = diffs.flatten(2).norm(p=2, dim=-1)  # shape: (N_real, N_gen)
                        # Cộng dồn với hệ số chia theo số lượng mẫu real
                        loss_discrepancy = loss_discrepancy + norms.sum() / real_vals.shape[0]

                    if gen_mask.sum() > 0:
                        gen_vals = a[gen_mask]   # shape: (N_gen, C, H, W)
                        if gen_vals.shape[0] > 1:
                            # Tính hiệu từng cặp trong gen_vals
                            diffs = gen_vals.unsqueeze(1) - gen_vals.unsqueeze(0)  # shape: (N_gen, N_gen, C, H, W)
                            norms = diffs.flatten(2).norm(p=2, dim=-1)  # shape: (N_gen, N_gen)
                            loss_robustness = loss_robustness + norms.sum() / gen_vals.shape[0]

            # In ra thông tin loss và trạng thái requires_grad cho debug (detach() để in giá trị)
            print(f"loss_cls: {loss_cls.detach()} \n"
                f"real_annotation_loss: {real_annotation_loss.detach()} \n"
                f"loss_discrepancy: {loss_discrepancy.detach()} \n"
                f"loss_robustness: {loss_robustness.detach()}")
            print(f"loss_cls.requires_grad: {loss_cls.requires_grad} - "
                f"real_annotation_loss.requires_grad: {real_annotation_loss.requires_grad} - "
                f"loss_discrepancy.requires_grad: {loss_discrepancy.requires_grad} - "
                f"loss_robustness.requires_grad: {loss_robustness.requires_grad}")

            # Tổng hợp các loss với hệ số tương ứng và tính trung bình nếu cần
            loss_total = loss_cls.mean() \
                        + lambda_0 * real_annotation_loss.mean() \
                        + lambda_1 * loss_discrepancy \
                        + lambda_2 * loss_robustness

            # Ghi log loss và các metric khác (nếu có)
            losses = {}
            losses[f"{prefix}_loss"] = loss_total.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(logits, sub_labels, k=1, reduction="none")
            log_loss_dict(diffusion, sub_t, losses)

            # Backward qua mp_trainer (ở chế độ train)
            if loss_total.requires_grad and prefix == "train":
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss_total * len(sub_batch) / len(batch))

        return losses


    #### every step 
    loss_epoch = 0
    acc_epoch = 0
    val_losses = []
    val_accuracies = []
    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        # print('step', step + resume_step)
        
        losses = forward_backward_log(datal, data) #losses for each batch: data = iter(datal)

        loss_epoch += losses['train_loss'].sum()
        acc_epoch += losses['train_acc@1'].sum()

        mp_trainer.optimize(opt)
        # calculate val_accuracy & loss in all of validation dataset
        if val_data is not None and not step % args.eval_interval:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
                    #forward_backward_log(val_datal, val_data, prefix="val")
                    val_loss, val_accuracy = validation_log(val_datal)
                    wandb.log({
                        "step": step + resume_step,
                        "val_acc": val_accuracy,
                        "val_loss": val_loss,
                    })
                    model.train()

        if not step % args.log_interval:
            print('step', step + resume_step)
            logger.dumpkvs()
            wandb.log({
                "step": step,
                "train_acc@1_10_step": losses['train_acc@1'].mean(),
                "train_loss_10_step": losses['train_loss'].mean(),
            })
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)

        if not (step+1) % len(datal): ## số batch: (len(datal))
            wandb.log({
                "epoch": (step+1)/(len(datal)),
                "train_acc@1": acc_epoch/(len(datal))/args.batch_size,
                "train_loss": loss_epoch/(len(datal))/args.batch_size,
            })
            loss_epoch = 0
            acc_epoch = 0
        
    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        save_dir = os.path.join(logger.get_dir(), "classifier")
        os.makedirs(save_dir, exist_ok=True)
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))

def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=True, ############################################
        iterations= 100001, # must be more than step from checkpoint
        lr=1e-4,
        weight_decay=0.0,
        anneal_lr=True,
        batch_size=8,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",#f"/kaggle/input/brats20-models-fold2/modelcls020000.pt",
        log_interval=10,
        eval_interval=1000,
        save_interval=10000,
        dataset='brats',
        max_L=1000,
        fold=2,
        transform=False,
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()


##### Bây giờ, có 2 hướng chính:
### Cách 1: Cải thiện chất lượng của cls - grad của cls sẽ tốt theo: Contrastive, Kernel-diversity (Vẫn dùng mask cũ)
### + Về kernel_diversity: Chỉ nên diversity ở những layer CNN đầu thôi, chứ còn về sau khi nó đã tổng hợp được nhiều thông tin thì ko cần bắt phải khác nhau.
### + Có thể sử dụng thêm LASSO regularization --> loại bỏ những nơ-ron ko cần thiết (dư thừa)
### Nhưng mà thực sự thì cls ở đây đang có 2 thái cực: nếu chỉ tập trung vào ảnh gốc ban đầu thì mask khởi tạo rất tốt, nếu muốn hướng dẫn grad thì chưa đủ thuyết phục.
### Vậy có nên train thêm 1 cls chỉ dựa trên ảnh ban đầu thui. Kết hợp giữa nó và cls trên trên ảnh noise để refined 
### Làm như vậy coarse mask sẽ chính xác hơn nhiều.

### Cách 2: Bản chất của DDIM và DDPM ở đây cũng chỉ là hỗ trợ cho grad của cls mà thôi, kiểu coarse mask sẽ được tinh chỉnh
### Trong khi DDIM sẽ heal nhẹ nhàng hơn, thì DDPM sẽ heal mãnh liệt hơn ở vùng nghi ngờ của nó.
### Vậy liệu có cách nào mask ngon hơn nhiều ko??? --> có thể train thêm mạng khác, hoặc cái gì đó, thay vì dùng đạo hàm tại thời điểm ban đầu (có thể kết hợp thêm các thời điểm khác) làm mask


'''
def forward_backward_log(data_load, data_loader, prefix="train"):
        try:
            batch, _, labels, masks = next(data_loader)
        except:
            data_loader = iter(data_load)
            batch, _, labels, masks = next(data_loader)

        # print('labels', labels)
        batch = batch.to(dist_util.dev())
        labels= labels.to(dist_util.dev())
        masks = masks.to(dist_util.dev())
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            # print(f"{prefix}: batch_shape: {batch.shape} - noise_levels: {t}")
            batch_0 = batch
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        ##### Ở đây đang cố thử hiệu chỉnh bằng 2 thành phần: Cross_entropy loss + mse(coarse_mask(grad của cls), groundtruth_mask) --> supervised DDPM, not  WSSS
        for i, (sub_batch_0,sub_batch, sub_labels, sub_masks, sub_t) in enumerate(
            split_microbatches(args.microbatch,batch_0, batch, labels, masks, t)
        ):
            #
            logits = model(sub_batch, timesteps=sub_t)

            #
            t_0 = th.randint(low=0, high=1, size=(sub_batch_0.shape[0],), device=dist_util.dev())
            ds_label = th.randint(low=0, high=1, size=(sub_batch_0.shape[0],), device=dist_util.dev())
            with th.enable_grad():
                sub_batch_0_detached = sub_batch_0.detach().requires_grad_(True)
                logits_0 = model(sub_batch_0_detached, t_0)
                #print("logits_0.requires_grad:", logits_0.requires_grad)
                log_probs = F.log_softmax(logits_0, dim=-1)
                #print("log_probs.requires_grad:", log_probs.requires_grad)
                selected = log_probs[range(len(logits_0)), ds_label.view(-1)]
                #print("selected.requires_grad:", selected.requires_grad)
                # Tính toán gradient của sub_batch_0
                x0_grad = th.autograd.grad(selected.sum(), sub_batch_0_detached)[0]
            grad_img = th.abs(th.sum(x0_grad, dim=1))  # từ (B,C,H,W) thành (B,H,W)
            coarse_mask_0 = min_max_scaler(grad_img)  # normalized coarse_mask 
            
            #coarse_mask_ = (th.ones(coarse_mask.shape, device=coarse_mask.device) - coarse_mask)
         
            loss = F.cross_entropy(logits, sub_labels, reduction="none") + F.mse_loss(coarse_mask_0, sub_masks, reduction="mean")

'''



'''
2 kịch bản:
 + Thiết kế train few-shot, cần phải chọn ra một vài tầm 100 mẫu đã có đủ annotation, train hy vọng sẽ làm cải thiện.
 + Train theo kiểu đạo hàm trên ảnh ban đầu khi tách ra thành các patch nên về giá trị trung bình của nó.
'''