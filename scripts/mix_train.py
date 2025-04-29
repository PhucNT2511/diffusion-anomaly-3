"""
Train a noised image mix on Brats20.
"""
import wandb
import argparse
import os
import sys
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import (
    BRATSDataset, 
    split_dataset_by_annotation_and_cluster,
    SubClusterBatchSampler
)
# from guided_diffusion.litsloader import LiTSDataset

import blobfile as bf
import torch as th
# from guided_diffusion.losses import FocalLoss
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader, Sampler


from torchvision import transforms
from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.train_util import visualize
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    mix_and_diffusion_defaults,
    create_mix_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict


def main():

    def dice_loss(pred, target, smooth=1e-8):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    def focal_loss(pred, target, alpha=0.8, gamma=2.0, eps=1e-8):
        pred = pred.clamp(eps, 1. - eps)  # tránh log(0)
        loss = -alpha * (1 - pred)**gamma * target * th.log(pred) \
            - (1 - alpha) * pred**gamma * (1 - target) * th.log(1 - pred)
        return loss.mean()


    def min_max_scaler(x):
        x_flat = x.reshape(x.shape[0], -1)
        x_min = th.min(x_flat, dim=1).values
        x_max = th.max(x_flat, dim=1).values
        scale = x_max - x_min
        x_normalize = (x - x_min[:, None, None]) / scale[:, None, None]
        return x_normalize

    
    args = create_argparser().parse_args()

    wandb.login(key="18867541319386f8b2e1362741174bd50968c3f3")
    wandb.init(
        project="brats-few-shot-mix",  # Replace with your project name
        config=args,  # Optionally log hyperparameters
    )

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_mix_and_diffusion(
        **args_to_dict(args, mix_and_diffusion_defaults().keys()),
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
        print(args.min_cluster_size)
        print( args.max_cluster_size)
        w_ds, a_ds, c_ds, subcs = split_dataset_by_annotation_and_cluster(
            ds, args.min_cluster_size, args.max_cluster_size
        )
        w_loader = DataLoader(w_ds, batch_size=args.batch_size, shuffle=True)
        a_loader = DataLoader(a_ds, batch_size=args.batch_size, shuffle=True)
        c_sampler = SubClusterBatchSampler(subcs, args.subclusters_per_batch)
        c_loader = DataLoader(c_ds, batch_sampler=c_sampler)
        w_iter, a_iter, c_iter = iter(w_loader), iter(a_loader), iter(c_loader)

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
            shuffle=False)
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

    logger.log("training mixed model...")

    def validation_log(val_data_load):
        ### only in clean images
        data_loader = iter(val_data_load)
        accuracies = []
        losses = []
        data_size = 0
        for data in data_loader:
            batch, _, labels, _, _, _ = data
            data_size += batch.shape[0]
            batch = batch.to(dist_util.dev())
            labels= labels.to(dist_util.dev())
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())
            for i, (sub_batch, sub_labels, sub_t) in enumerate(
                split_microbatches(args.microbatch, batch, labels, t)
            ):
            
                logits, saliency = model(sub_batch, sub_batch, timesteps=sub_t)
            
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
    lambda_0 = args.lambda_0
    lambda_1 = args.lambda_1
    lambda_2 = args.lambda_2

    def forward_backward_log(
        w_loader, w_iter,
        a_loader, a_iter,
        c_loader, c_iter,
        prefix="train"
    ):
        """
        Sequentially fetch batches from weak, annotated, and clustered loaders, compute individual losses,
        print debug info, log, and backprop as in the original implementation.
        Returns loss dict and updated iterators.
        """
        dev = dist_util.dev()

        # --- 1) Weak (classification) batch ---
        try:
            w_batch, _, w_labels, _, w_exist, _ = next(w_iter)
        except StopIteration:
            w_iter = iter(w_loader)
            w_batch, _, w_labels, _, w_exist, _ = next(w_iter)
        w_batch, w_labels = w_batch.to(dev), w_labels.to(dev)

        # Noise sampling if enabled
        if args.noised:
            t_w, _ = schedule_sampler.sample(w_batch.size(0), dev)
            w_input = diffusion.q_sample(w_batch, t_w)
        else:
            t_w = th.zeros(w_batch.size(0), dtype=th.long, device=dev)
            w_input = w_batch

        # Classification forward
        logits, _ = model(w_input, w_batch, timesteps=t_w)
        loss_cls = F.cross_entropy(logits, w_labels, reduction="mean")

        # --- 2) Annotation batch ---
        try:
            a_batch, _, _, a_masks, a_exist, _ = next(a_iter)
        except StopIteration:
            a_iter = iter(a_loader)
            a_batch, _, _, a_masks, a_exist, _ = next(a_iter)
        a_batch, a_masks, a_exist = a_batch.to(dev), a_masks.to(dev), a_exist.to(dev)

        # Annotation forward
        _, sal_ann = model(a_batch, a_batch, timesteps=None)
        # Real annotation loss
        loss_anno = th.tensor(0., device=dev)
        cnt_anno = a_exist.sum().item()
        if cnt_anno > 0:
            mask_ann = a_masks.unsqueeze(1).float()
            loss_anno = F.binary_cross_entropy(sal_ann[a_exist==1], mask_ann[a_exist==1])

            '''
            bce = F.binary_cross_entropy_with_logits(logits, mask)
            # Sigmoid để tính dice
            probs = torch.sigmoid(logits)
            inter = (probs * mask).sum(dim=(2,3))
            union = probs.sum(dim=(2,3)) + mask.sum(dim=(2,3))
            dice = 1 - (2*inter + 1e-6) / (union + 1e-6)
            loss_dice = dice.mean()
            loss = bce + loss_dice

            '''

        # --- 3) Clustered batch ---
        try:
            c_batch, _, _, _, c_exist, c_cluster = next(c_iter)
        except StopIteration:
            c_iter = iter(c_loader)
            c_batch, _, _, _, c_exist, c_cluster = next(c_iter)
        c_batch, c_exist, c_cluster = c_batch.to(dev), c_exist.to(dev), c_cluster.to(dev)

        # Cluster forward
        _, sal_cl = model(c_batch, c_batch, timesteps=None)

        # Discrepancy loss
        loss_discrepancy = th.tensor(0., device=dev)
        for c in c_cluster.unique():
            real_mask = (c_cluster==c) & (c_exist==1)
            gen_mask  = (c_cluster==c) & (c_exist==0)
            n_real, n_gen = real_mask.sum().item(), gen_mask.sum().item()
            if n_real>0 and n_gen>0:
                real_vals = sal_cl[real_mask]
                gen_vals  = sal_cl[gen_mask]
                diffs = real_vals.unsqueeze(1) - gen_vals.unsqueeze(0)
                norms = th.sqrt(diffs.pow(2).mean(dim=(2,3)))
                loss_discrepancy += norms.sum() / (n_real * n_gen)

        # Robustness loss
        loss_robustness = th.tensor(0., device=dev)
        for c in c_cluster.unique():
            gen_mask = (c_cluster==c) & (c_exist==0)
            n_gen = gen_mask.sum().item()
            if n_gen>1:
                vals = sal_cl[gen_mask]
                diffs = vals.unsqueeze(1) - vals.unsqueeze(0)
                norms = th.sqrt(diffs.pow(2).mean(dim=(2,3)))
                loss_robustness += norms.sum() / (n_gen * n_gen)

        # Debug prints
        print(f"loss_cls: {loss_cls.detach()} \n"
            f"real_annotation_loss: {loss_anno.detach()} \n"
            f"loss_discrepancy: {loss_discrepancy.detach()} \n"
            f"loss_robustness: {loss_robustness.detach()}")
        print(f"loss_cls.requires_grad: {loss_cls.requires_grad} - "
            f"real_annotation_loss.requires_grad: {loss_anno.requires_grad} - "
            f"loss_discrepancy.requires_grad: {loss_discrepancy.requires_grad} - "
            f"loss_robustness.requires_grad: {loss_robustness.requires_grad}")

        # Total loss
        loss_total = (loss_cls
                    + lambda_0 * loss_anno
                    + lambda_1 * loss_discrepancy
                    + lambda_2 * loss_robustness)

        # Log and backward
        losses = {}
        losses[f"{prefix}_loss"] = loss_total.detach()
        losses[f"{prefix}_acc@1"] = compute_top_k(logits, w_labels, k=1, reduction="none")
        log_loss_dict(diffusion, t_w, losses)

        if loss_total.requires_grad and prefix=="train":
            mp_trainer.zero_grad()
            mp_trainer.backward(loss_total)
            mp_trainer.step()

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
        
        losses = forward_backward_log(
            w_loader, w_iter, a_loader, a_iter, c_loader, c_iter
        )

        loss_epoch += losses['train_loss'].sum()
        acc_epoch += losses['train_acc@1'].sum()

        mp_trainer.optimize(opt)
        # calculate val_accuracy & loss in all of validation dataset
        if val_data is not None and not step % args.eval_interval:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
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
        subclusters_per_batch=2,
        min_cluster_size=3,
        max_cluster_size=10,
        lambda_0=1.0,
        lambda_1=0.1,
        lambda_2=1.0,
    )
    defaults.update(mix_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
