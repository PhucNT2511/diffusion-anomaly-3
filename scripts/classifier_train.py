"""
Train a noised image classifier on ImageNet.
"""

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
    classifier_scale = 100
    '''
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
            batch_size=args.batch_size,
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

        for i, (sub_batch_0,sub_batch, sub_labels, sub_masks, sub_t) in enumerate(
            split_microbatches(args.microbatch,batch_0, batch, labels, masks, t)
        ):
            #
            logits = model(sub_batch, timesteps=sub_t)

            #
            t_0 = th.randint(low=0, high=1, size=(sub_batch_0.shape[0],), device=dist_util.dev())
            ds_label = th.randint(low=0, high=1, size=(sub_batch_0.shape[0],), device=dist_util.dev())
            logits_0 = model(sub_batch_0, t_0)
            log_probs = F.log_softmax(logits_0, dim=-1)

            # Chọn giá trị từ log_probs theo các nhãn ngẫu nhiên
            selected = log_probs[range(len(logits)), ds_label.view(-1)]

            # Tính toán gradient của sub_batch_0
            x0_grad = th.autograd.grad(selected.sum(), sub_batch_0)[0]
            grad_img = th.abs(th.sum(x0_grad, dim=1))  # từ (B,C,H,W) thành (B,H,W)
            coarse_mask = min_max_scaler(grad_img)  # mask
            soft_mask = th.sigmoid((0.4 - coarse_mask) * 1000)  # ngưỡng 0.4 - sigmoid 1/(1+e^-t) 
         
            loss = F.cross_entropy(logits, sub_labels, reduction="none") + F.mse_loss(soft_mask,sub_masks, reduction="mean")
            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )
            # losses[f"{prefix}_acc@2"] = compute_top_k(
            #     logits, sub_labels, k=2, reduction="none"
            # )
            # print('acc', losses[f"{prefix}_acc@1"])
            log_loss_dict(diffusion, sub_t, losses)
            loss = loss.mean()
#             if prefix=="train":
#                 pass
# #                 viz.line(X=th.ones((1, 1)).cpu() * step, Y=th.Tensor([loss]).unsqueeze(0).cpu(),
# #                      win=loss_window, name='loss_cls',
# #                      update='append')

#             else:

#                 output_idx = logits[0].argmax()
#                 print('outputidx', output_idx)
#                 output_max = logits[0, output_idx]
#                 print('outmax', output_max, output_max.shape)
#                 output_max.backward()
#                 saliency, _ = th.max(sub_batch.grad.data.abs(), dim=1)
#                 print('saliency', saliency.shape)
# #                 viz.heatmap(visualize(saliency[0, ...]))
# #                 viz.image(visualize(sub_batch[0, 0,...]))
# #                 viz.image(visualize(sub_batch[0, 1, ...]))
#                 th.cuda.empty_cache()


            if loss.requires_grad and prefix=="train":
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

        return losses

    correct=0; total=0
    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        # print('step', step + resume_step)
        
        losses = forward_backward_log(datal, data)

        correct+=losses["train_acc@1"].sum()
        total+=args.batch_size
        acctrain=correct/total

        mp_trainer.optimize(opt)
        if val_data is not None and not step % args.eval_interval:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_datal, val_data, prefix="val")
                    val_loss, val_accuracy = validation_log(val_datal)
                    print(f"Validation loss: {val_loss} - Validation accuracy: {val_accuracy}")
                    model.train()

        if not step % args.log_interval:
            print('step', step + resume_step)
            logger.dumpkvs()
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)

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
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"classifier/model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"classifier/opt{step:06d}.pt"))

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
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=True,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=1000,
        save_interval=10000,
        dataset='brats',
        max_L=1000,
        fold=1,
        transform=False,
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
