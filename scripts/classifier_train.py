"""
Train a noised image classifier on ImageNet.
"""
import wandb
import argparse
import os
import sys
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.camelyonloader import CAMELYONDataset
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
    wandb.login(key="18867541319386f8b2e1362741174bd50968c3f3")
    args = create_argparser().parse_args()

    wandb.init(
        project="noised-image-classifier",  # Replace with your project name
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

    if args.dataset == 'camelyon':
        print("Training on CAMELYON-16 dataset")
        ds = CAMELYONDataset(mode="train", test_flag=False, transforms=transform, model='classifier')
        datal = th.utils.data.DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=True)
        data = iter(datal)
        print('train_datal_len: ',len(datal))

    # elif args.dataset == 'lits':
    #     print("Training on LiTS dataset")

    #     ds = LiTSDataset(args.data_dir, mode="train", test_flag=False)
    #     datal = th.utils.data.DataLoader(
    #         ds,
    #         batch_size=args.batch_size,
    #         shuffle=True)
    #     data = iter(datal)

    try:
        val_ds = CAMELYONDataset(mode="val", test_flag=False, model='classifier')
        val_datal = th.utils.data.DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=True)
        val_data = iter(val_datal)
        print('val_datal_len: ',len(val_datal))
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

        return np.mean(losses), np.mean(accuracies) # Transform to numpy
    
    def forward_backward_log(data_load, data_loader, prefix="train"):
        try:
            batch, _, labels, _ = next(data_loader)
        except:
            data_loader = iter(data_load)
            batch, _, labels, _ = next(data_loader)

        # print('labels', labels)
        batch = batch.to(dist_util.dev())
        labels= labels.to(dist_util.dev())
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            # print(f"{prefix}: batch_shape: {batch.shape} - noise_levels: {t}")
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        ## Calculate for 1 batch
        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
          
            logits = model(sub_batch, timesteps=sub_t)
         
            loss = F.cross_entropy(logits, sub_labels, reduction="none")
            losses = {}
            # calculate loss & acc@1 in 1 batch of val and train set
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )
            # losses[f"{prefix}_acc@2"] = compute_top_k(
            #     logits, sub_labels, k=2, reduction="none"
            # )
            # print('acc', losses[f"{prefix}_acc@1"])
            log_loss_dict(diffusion, sub_t, losses)
            loss = loss.mean() ## mean loss, but don't return this value
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

    #### every step 
    for step in range(args.iterations - resume_step):
        # logger.logkv sẽ lưu lại toàn bộ giá trị, và thực chất là lưu lại của vòng lặp trước đó
        logger.logkv("step", step + resume_step)
        # ở đây cũng chỉ ghi thêm "samples" thứ bn thôi, còn các giá trị khác của loss đã lưu hết ở hàm log_loss_dict
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
    
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        # print('step', step + resume_step)
        
        losses = forward_backward_log(datal, data) #losses for each batch: data = iter(datal)
        mp_trainer.optimize(opt)
        # calculate val_accuracy & loss in all of validation dataset - sau 1000 steps sẽ in ra kết quả evaluate
        if val_data is not None and not step % args.eval_interval:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_datal, val_data, prefix="val")
                    val_loss, val_accuracy = validation_log(val_datal)
                    ## Sau 1000 steps wandb lưu lại thông số của validate
                    wandb.log({
                        "step": step + resume_step,
                        "val_acc": val_accuracy,
                        "val_loss": val_loss,
                    })
                    model.train()

        if not step % args.log_interval: # cứ 10 steps thì in ra một lần
            print('step', step + resume_step)
            logger.dumpkvs() #logger.logkv đã lưu rồi thì logger.dumpkvs() sẽ in ra giá trị cuối cùng được lưu lại trong logger
            ## wandb lưu lại sau 10 steps các thông số của training
            wandb.log({
                "step": step + resume_step,
                "train_acc@1": losses["train_acc@1"].mean(),
                "train_loss": losses["train_loss"].mean(),
            })
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
        noised=True,
        iterations=100000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=True,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=1000, # sau 1000 steps sẽ in ra kết quả evaluate
        save_interval=10000, # sau 10000 steps sẽ lưu lại một lần
        dataset='camelyon',
        max_L=1000,
        transform=False,
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()