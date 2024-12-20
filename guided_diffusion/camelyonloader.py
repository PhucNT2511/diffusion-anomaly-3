import os
import torch 
import numpy as np
import torch.nn.functional as F
import pickle
import pandas as pd


#data_path = '/kaggle/input/camelyon'
data_path = '/kaggle/input/camelyon16/data'
#data_path = 'D:/medical_DF/data_camelyon

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

class CAMELYONDataset(torch.utils.data.Dataset):
    def __init__(self, mode="train", test_flag = False, transforms=None, model = "unet"):
    
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

        if (model =="classifier"):
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
        
        if test_flag==True:
            paths = os.listdir(data_path + '/test')  # Lấy danh sách các file/thư mục
            for path in paths:
                full_path = os.path.join(data_path + '/test', path)  # Kết hợp đường dẫn đầy đủ
                self.datapaths.append(full_path)

    def __getitem__(self, idx):
        data = np.load(self.datapaths[idx],allow_pickle = True).item()
        image = np.array(data['image'])
        mask = np.array(data['mask'])
        
        ## 
        image = np.transpose(image, [2, 0, 1])
        image = normalize(image)

        label = 1 if np.sum(mask) > 0 else 0
         
        ####################### Init cond = None
        cond = {}
        cond['y'] = label 

        if self.transforms:
            image = self.transforms(torch.Tensor(image))

        return np.float32(image), cond, label, np.float32(mask)

    def __len__(self):
        return len(self.datapaths)