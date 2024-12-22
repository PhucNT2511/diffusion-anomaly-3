import os
import torch 
import numpy as np
import torch.nn.functional as F
import pickle
import pandas as pd
import matplotlib.pyplot as plt


#data_path = '/kaggle/input/camelyon'
data_path = '/kaggle/input/camelyon16/data'
#data_path = 'D:/medical_DF/data_camelyon

# Hàm xoay ảnh và mask theo các góc: 0°, 90°, 180°, 270°
def rotate_image(image, mask):
    # Tạo một danh sách chứa các ảnh và mask xoay ở 4 góc: 0°, 90°, 180°, 270°
    images = [image]
    masks = [mask]
    
    # Xoay 90 độ
    images.append(np.rot90(image, k=1))
    masks.append(np.rot90(mask, k=1))
    
    # Xoay 180 độ
    images.append(np.rot90(image, k=2))
    masks.append(np.rot90(mask, k=2))
    
    # Xoay 270 độ
    images.append(np.rot90(image, k=3))
    masks.append(np.rot90(mask, k=3))
    
    return images, masks

def normalize(image):
    """Basic min max scaler.
    """
    min_ = np.min(image)
    max_ = np.max(image)
    # if scale == 0: 
    # return 0 --> in ra số thứ tự của ảnh đó trong file [idx]
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

class CAMELYONDataset(torch.utils.data.Dataset):
    def __init__(self, mode="train", test_flag = False, transforms=None, model = "unet"):
        self.model = model
        self.mode = mode
    
        super().__init__()
        self.transforms = transforms
        if self.transforms:
            print("Transform for data augmentation.")
        else:
            print("No data augmentation")
        self.datapaths = []
        
        if (model =="unet"):
            paths = os.listdir(data_path +'/unet')
            for path in paths:
                full_path = os.path.join(data_path + '/unet', path)  # Kết hợp đường dẫn đầy đủ
                self.datapaths.append(full_path)

        elif (model =="classifier"):
            if (mode == "train"):
                paths = os.listdir(data_path +'/classifier/train')
                for path in paths:
                    full_path = os.path.join(data_path + '/classifier/train', path)  # Kết hợp đường dẫn đầy đủ
                    self.datapaths.append(full_path)    
            if (mode == "val"):
                paths = os.listdir(data_path +'/classifier/val')
                for path in paths:
                    full_path = os.path.join(data_path + '/classifier/val', path)  # Kết hợp đường dẫn đầy đủ
                    self.datapaths.append(full_path)  
        
        elif test_flag==True:
            paths = os.listdir(data_path + '/test')  # Lấy danh sách các file/thư mục
            for path in paths:
                full_path = os.path.join(data_path + '/test', path)  # Kết hợp đường dẫn đầy đủ
                self.datapaths.append(full_path)

    def __getitem__(self, idx):
        data = np.load(self.datapaths[idx],allow_pickle = True).item()
        image = np.array(data['image'])
        mask = np.array(data['mask'])
        

        if self.model == "classifier" and self.mode =="train":
            # Xoay ảnh và mask theo các góc 0°, 90°, 180°, 270°
            images, masks = rotate_image(image, mask)

            # Chuyển các ảnh về định dạng (C, H, W) và chuẩn hóa
            images = [np.transpose(im, [2, 0, 1]) for im in images]  # Chuyển từ (H, W, C) -> (C, H, W)
            images = [irm_min_max_preprocess(im) for im in images]  # Chuẩn hóa tất cả các ảnh

            # Tạo label cho tất cả các mask
            labels = [1 if np.sum(mask) > 0 else 0 for mask in masks]

            ####################### Init cond = None
            cond = {'y': labels}

            if self.transforms:
                # Chuyển mỗi ảnh thành Tensor và áp dụng các transform nếu có
                images = [self.transforms(torch.Tensor(image)) for image in images]

            return [np.float32(image) for image in images], cond, labels, [np.float32(mask) for mask in masks]


         ##
        image = np.transpose(image, [2, 0, 1])
        image = irm_min_max_preprocess(image)

        label = 1 if np.sum(mask) > 0 else 0
         
        ####################### Init cond = None
        cond = {}
        cond['y'] = label 
        if self.transforms:
            image = self.transforms(torch.Tensor(image))

        return np.float32(image), cond, label, np.float32(mask)

    def __len__(self):
        return len(self.datapaths)