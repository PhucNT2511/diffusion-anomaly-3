import pandas as pd
import os
import torch
import torch.utils.data as data
import imageio
import numpy as np
import torch.nn.functional as F
import pickle
from torchvision import datasets, models, transforms

def normalize(image):
    """Basic min max scaler.
    """
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
    return image

def irm_min_max_preprocess(image, low_perc=1, high_perc=99):
    """Main pre-processing function used for the challenge (seems to work the best).
    Remove outliers voxels first, then min-max scale.
    1% -- 99%
    Warnings
    --------
    This will not do it channel wise!!
    """

    non_zeros = image > 0
    if non_zeros.sum() > 0:
        low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
        image = np.clip(image, low, high)
        image = normalize(image)
    return image

def min_max_scaler(image):
    return (image - image.min()) / image.max()

def binarize(img):
    return np.where(img > 0, 1.0, 0.0)
    

directory = ['brats21-dataset', 'brats21-dataset2']

############# 4 layers + groundtruth mask (seg)
###flair[k], t1[k], t1ce[k], t2[k] - my order
###seqtypes = ['flair','t1', 't1ce', 't2', 'seg']

for i1 in directory:
    for i2 in os.listdir(i1):  #group
        i12 = os.path.join(i1,i2)
        for i3 in os.listdir(i12): #patient
            i123 = os.path.join(i12,i3) 
            for i4 in os.listdir(i123): #slice - 4 types of img + seg
                i1234 = os.path.join(i123,i4) 
                data = np.load(i1234, allow_pickle = True)
                # Load image and mask

                image = data['image'].astype(np.float32)  # Convert to float32
                mask = data['mask'].astype(np.float32)  # Convert to float32

                ###
                image = image.reshape(image.shape[0],image.shape[2],image.shape[1])
                mask = mask.reshape(mask.shape[1],mask.shape[0])

                if mask.shape[0] < 256:

                    # Process image (4 channels)
                    for i in range(image.shape[0]):
                        if np.sum(image[i]) > 0:
                            image[i] = min_max_scaler(image[i])
                            # image[i] = irm_min_max_preprocess(image[i])
    
                    if np.sum(mask) > 0:
                        mask = binarize(mask)
    
                    # Padding (8 pixels on each side)
                    image = np.pad(image, ((0, 0), (8, 8), (8, 8)), mode='constant', constant_values=0)
                    mask = np.pad(mask, ((8, 8), (8, 8)), mode='constant', constant_values=0)
    
                    np.savez(i1234, image=image, mask=mask)
                    
                elif mask.shape[0] > 256:
                    
                    image = image[:, 8:-8, 8:-8]  # Giữ nguyên số kênh, chỉ cắt bớt chiều cao và chiều rộng
                    mask = mask[8:-8, 8:-8]       # Cắt bớt mask theo cùng một cách
    
                    np.savez(i1234, image=image, mask=mask)

                
        


################ cảm giác chỉ là chuẩn hóa ảnh và sau đó lưu vào các file với tên tuong ứng --> xem xét bỏ qua
################# KO CẦN DÙNG FILE NÀY