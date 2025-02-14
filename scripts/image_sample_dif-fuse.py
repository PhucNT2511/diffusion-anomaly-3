import sys
# put your path here
#sys.path.extend(['/disk/scratch2/alessandro/new_code/Dif-fuse'])
sys.path.append("..")
sys.path.append(".")
import matplotlib.pyplot as plt
import argparse
import cv2

import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.brain_datasets import *
import torchvision
import torch
import torch.nn as nn
from torchvision import utils
from torch.utils.data import DataLoader
import nibabel as nib

def load_niftii_file(file_path):
    image = nib.load(file_path)
    niifti_data = image.get_fdata()
    niifti_data = niifti_data.astype(np.float32)
    return niifti_data

def thresholdf(x, percentile, independent):
    if independent ==1:
        a = x * (x.numpy() > np.percentile(x, percentile, axis = (-2,-1), keepdims = True))
    else:
        a = x * (x.numpy() > np.percentile(x, percentile, axis = (-3,-2,-1), keepdims = True))
    return a

def clean(saliency, threshold, independent):

    saliency = thresholdf(saliency,threshold, independent)
    return saliency

def normalise(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

kernel3 = np.ones((3, 3), np.uint8)
kernel5 = np.ones((5, 5), np.uint8)
def main():
    args = create_argparser().parse_args()

    print(vars(args))

    dist_util.setup_dist()
    logger.configure(experiment_name=args.experiment_name)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else "cpu"
    )
    model.to(device)

    if args.use_fp16:
        model.convert_to_fp16()
    if args.gpus > 1:
        model = nn.DataParallel(model)

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path)
    )

    model.eval()

    val_dataset = BRATSDatasetSaliency(
            saliency_root_folder_filepath= '/kaggle/input/saliency-maps-fold-1/diffusion-anomaly-3/saliency_maps',
            fold=args.fold,
            transform = None,
            only_positive = True,
            only_negative = False)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    logger.log("sampling...")
    all_images = []

    noise_level = 500
    threshold = 90
    range_t = -1
    kernel = 5
    independent = 0

    for i, (image,_,_,sal, ids) in enumerate(val_loader):
            if i*args.batch_size < args.start_point:
                continue
            elif i*args.batch_size >= args.end_point:
                break
            sample_fn = (
                diffusion.diffuse_loop_forward_backward
            )
            sal = clean(sal, threshold=threshold, independent = independent)
            mask = sal.to(torch.float).to(device)


            if kernel>0:
                for j in range(mask.shape[0]):
                    for each_slice in range(mask.shape[1]):
                        # print(i, each_slice)
                        mask[j, each_slice, :, :] = torchvision.transforms.functional.gaussian_blur(
                            mask[j, each_slice, :, :].view(1, mask.shape[2], mask.shape[3]), kernel_size=kernel).view(
                            mask.shape[2], mask.shape[3])


            mask[mask > 0] = 1
            mask[mask == 0] = 0
            mask = mask.to(device)

            image = image.to(torch.float).to(device)
            ###################### Cái save ở ddaaay chỉ là load ra thôi, chứ chưa có tác dụng gì
            for k, level in enumerate(['flair', 't1', 't2', 't1ce']):
                utils.save_image((image[:, k, :, :]).unsqueeze(1),os.path.join(logger.get_dir(), f'batch{i}_threshold_{threshold}_{level}_image.png'), nrow=4)
            ############ reconstructed, sampled, original
            rec, sample, orig= sample_fn(
                model = model,
                mask = mask,
                shape = (args.batch_size, 4, args.image_size, args.image_size),
                img =image.to(device),
                clip_denoised=args.clip_denoised,
                noise_level=noise_level,
                cond_fn=None,
                device=device,
                range_t =range_t
            )

            
            sample_img = sample.to(torch.float)

            out_path = os.path.join(logger.get_dir(), 'images')
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            for j in range(sample_img.shape[0]):
                out_path_img_sample = os.path.join(logger.get_dir(),
                                                    f"images/batch{i}_noiselevel_{noise_level}_threshold_{threshold}_kernelsize_{kernel}_ind_{independent}_{ids[j][40:-4]}_sample.npy")
                with open(out_path_img_sample, 'wb') as f:
                    np.save(f, np.array(sample_img[j, :, :, :]))
                print(f'Process {max(i-1,0)*args.batch_size+j+1} images completely!')
            '''
            fig = plt.figure(figsize=(11,11))
            for j in range(args.batch_size):
                plt.subplot(4, 4, j + 1)
                plt.grid(visible=False)
                plt.axis('off')
                plt.imshow(erode[j,:,:,:].squeeze(0), interpolation='none', cmap="Reds")
            ################## Lưu anomaly maps cho cả batch
            out_path_img_anomaly = os.path.join(logger.get_dir(),
                                                    f"images/batch{i}_noiselevel_{noise_level}_threshold_{threshold}_kernelsize_{kernel}_ind_{independent}_anomaly_map.png")
            plt.savefig(out_path_img_anomaly)
            plt.close(fig)
            

            fig = plt.figure(figsize=(11, 11))
            for j in range(args.batch_size):
                plt.subplot(4, 4, j + 1)
                plt.grid(visible=False)
                plt.axis('off')
                plt.imshow((orig_img[j, 0, :, :]).detach().cpu().numpy(), cmap=plt.cm.bone)
                plt.imshow(erode[j, 0, :, :], interpolation='none', alpha=0.5, cmap="Reds")

            ######################## Lưu anomaly map đè lên original image
            out_path_img_anomaly_overlayed = os.path.join(logger.get_dir(),
                                                    f"images/batch{i}_noiselevel_{noise_level}_threshold_{threshold}_ranget_{range_t}_kernelsize_{kernel}_anomaly_overlayed.png")

            plt.savefig(out_path_img_anomaly_overlayed)
            plt.close(fig)
            '''



def create_argparser():
    defaults = dict(
        clip_denoised=True,
        experiment_name='dif-fuse_sampling',
        gpus=1,
        num_samples=10000,
        batch_size=16,
        use_ddim=True,
        model_path="", ############# các path này phải lấy kỹ, theo fold
        classifier_path="", ################ path này lấy kỹ, theo fold
        fold = 1,
        start_point = 0,
        end_point = 500,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
