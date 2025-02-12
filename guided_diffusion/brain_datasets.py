import pandas as pd
import os
import torch
import torch.utils.data as data
import imageio
import numpy as np
import torch.nn.functional as F
import pickle

#Hai hàm này ko có vấn đề gì
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

class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 mode="train", 
                 fold=1,
                 transforms=None,
                 only_positive = False, ##Hai cờ này ý là: chỉ lấy dương/âm hay cả 2.
                 only_negative = False):
        
        super().__init__()
        self.datapaths = []
        self.transforms = transforms
        if self.transforms:
            print("Transform for data augmentation.")
        else:
            print("No data augmentation")

        self.only_positive = only_positive
        self.only_negative = only_negative

        
        ## train cả classifier và train model đều trên cùng một tập dữ liệu ?? (như vậy thì mất cân bằng nhãn cho classifier hả)
        data_split = np.load('/kaggle/working/diffusion-anomaly-3/data/brats/data_split.npz', allow_pickle=True)
        meta_data_df = pd.read_csv('/kaggle/working/diffusion-anomaly-3/data/brats/meta_data.csv')
        volume_ids = data_split[f'{mode}_folds'].item()[f'fold_{fold}']
        
        ############## Cần dẫn link cho 2 nhóm, một nhóm toàn link positive, 1 nhóm toàn link negative. Hoặc dùng chung nhưng phải có lable
        if self.only_positive:
            self.datapaths = meta_data_df[(meta_data_df['volume'].isin(volume_ids)) & (meta_data_df['label'] == 1)]['path'].values
        elif self.only_negative:
            self.datapaths = meta_data_df[(meta_data_df['volume'].isin(volume_ids)) & (meta_data_df['label'] == 0)]['path'].values
        else:
            self.datapaths = meta_data_df[meta_data_df['volume'].isin(volume_ids)]['path'].values

        print(f'Number of {mode} data: {len(self.datapaths)}')
        

    def __getitem__(self, idx):
        data = np.load(self.datapaths[idx])
        image = data['image']
        image = image[[1, 2, 3, 0], :, :]
        for i in range(image.shape[0]):
            image[i] = irm_min_max_preprocess(image[i])
        mask = data['mask']
        padding_image = np.zeros((4, 256, 256))
        padding_image[:, 8:-8, 8:-8] = image
        padding_mask = np.zeros((256, 256))
        padding_mask[8:-8, 8:-8] = mask
        label = 1 if np.sum(mask) > 0 else 0

        if self.transforms:
            padding_image = self.transforms(torch.Tensor(padding_image))
        return torch.tensor(padding_image, dtype= torch.float32), label, torch.tensor(padding_mask, dtype = torch.float32), self.datapaths[idx]

    def __len__(self):
        return len(self.datapaths)

#################### CSDL khi đã tạo được saliency

class BRATSDatasetSaliency(torch.utils.data.Dataset):
    def __init__(
            self,
            saliency_root_folder_filepath = '', # folder saliency
            mode="test",
            fold=1, 
            transform = None,
            only_positive = False,
            only_negative = False):
        
        super(BRATSDatasetSaliency, self).__init__()
        self.datapaths = []
        self.saliency_root_folder_filepath = saliency_root_folder_filepath
        self.only_positive = only_positive
        self.only_negative = only_negative
        self.transform = transform

        data_split = np.load('/kaggle/working/diffusion-anomaly-3/data/brats/data_split.npz', allow_pickle=True)
        meta_data_df = pd.read_csv('/kaggle/working/diffusion-anomaly-3/data/brats/meta_data.csv')
        volume_ids = data_split[f'{mode}_folds'].item()[f'fold_{fold}']
        
        ############## Cần dẫn link cho 2 nhóm, một nhóm toàn link positive, 1 nhóm toàn link negative. Hoặc dùng chung nhưng phải có lable
        if self.only_positive:
            self.datapaths = meta_data_df[(meta_data_df['volume'].isin(volume_ids)) & (meta_data_df['label'] == 1)]['path'].values
        elif self.only_negative:
            self.datapaths = meta_data_df[(meta_data_df['volume'].isin(volume_ids)) & (meta_data_df['label'] == 0)]['path'].values
        else:
            self.datapaths = meta_data_df[meta_data_df['volume'].isin(volume_ids)]['path'].values

    def __len__(self): ## len như bthg
        return len(self.datapaths)

    def __getitem__(self, idx): ## item vẫn là idx

        data = np.load(self.datapaths[idx])
        image = data['image']
        image = image[[1, 2, 3, 0], :, :]
        for i in range(image.shape[0]):
            image[i] = irm_min_max_preprocess(image[i])
        mask = data['mask']
        padding_image = np.zeros((4, 256, 256))
        padding_image[:, 8:-8, 8:-8] = image
        padding_mask = np.zeros((256, 256))
        padding_mask[8:-8, 8:-8] = mask
        label = 1 if np.sum(mask) > 0 else 0

        if self.transform:
            padding_image = self.transform(torch.Tensor(padding_image))

        raw_sal = []
        for level in ['flair', 't1', 't2', 't1ce']:
            sal = os.path.join(self.saliency_root_folder_filepath, self.datapaths[idx][40:-4] + level + '.png')
            sal = imageio.imread(sal)
            sal = torch.tensor(sal, dtype=torch.float32)
            sal = (sal/ torch.max(sal))
            raw_sal.append(sal)
        sal = torch.stack(raw_sal)

        return torch.tensor(padding_image, dtype=torch.float32), label, torch.tensor(padding_mask, dtype=torch.float32), sal, self.datapaths[idx]
        ## chỉ cần quan tâm tới đầu ra:
        ## image, label, seg (groundtruth_mask), saliency (được generate ra), path

#################### cần phân biệt rõ các tập này đang để làm gì? tập only_positive và negative
########3 flair ở đây chỉ để tìm link thui, ko có tác dụng gì nhiều


## Thực ra không khác gì cả, chỉ là tạo ra thêm một trường sal lấy ra từ folder
'''
class BRATSDatasetSaliency(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_root_folder_filepath, # folder dataset
            saliency_root_folder_filepath, # folder saliency
            df_path,
            transform = None,
            only_positive = False,
            only_negative = False,
            only_flair = False

    ):
        super(BRATSDatasetSaliency, self).__init__()
        self.dataset_root_folder_filepath = dataset_root_folder_filepath 
        self.saliency_root_folder_filepath = saliency_root_folder_filepath
        self.df_path = df_path
        self.only_positive = only_positive
        self.only_negative = only_negative
        self.transform = transform
        self.only_flair = only_flair

        self.meta_data_data_frame = pd.read_csv(
            self.df_path, encoding="ISO-8859-1"
        )
        if self.only_positive:
            self.meta_data_data_frame = self.meta_data_data_frame[self.meta_data_data_frame['label']==1]
        if self.only_negative:
            self.meta_data_data_frame = self.meta_data_data_frame[self.meta_data_data_frame['label']==0]

        self.sample_idx_to_scan_path_and_label = []

        self.sample_idx_to_scan_path_and_label = [
            (row["flair"], row["label"])  # Note we are using AIS lesion label here.
            for idx, row in self.meta_data_data_frame.iterrows()
        ]


    def __len__(self): ## len như bthg

        return len(self.sample_idx_to_scan_path_and_label)

    def __getitem__(self, item): ## item vẫn là idx
        x_path, y_sample = self.sample_idx_to_scan_path_and_label[item]  # example: 1085

        raw_image = []
        raw_sal = []
        for level in ['flair', 't1', 't2', 't1ce']:
            im = os.path.join(self.dataset_root_folder_filepath, x_path[:-9] + level + '.png')
            sal = os.path.join(self.saliency_root_folder_filepath, x_path[:-9] + level + '.png')
            image = imageio.imread(im)
            image = torch.tensor(image, dtype=torch.float32)
            sal = imageio.imread(sal)
            sal = torch.tensor(sal, dtype=torch.float32)
            if self.transform is not None:
                image = self.transform(image)
            image = (image / torch.max(image))
            sal = (sal/ torch.max(sal))
            raw_image.append(image)
            raw_sal.append(sal)
        ## Lấy luôn ở đây, bỏ phần trên đi   
        im = torch.stack(raw_image)
        sal = torch.stack(raw_sal)

        seg = os.path.join(self.dataset_root_folder_filepath[:-6] + 'segs', x_path[:-9] + 'seg' + '.png')
        seg = imageio.imread(seg)
        seg = torch.tensor(seg).unsqueeze(0)
        seg = (seg / torch.max(seg))
        if torch.max(seg) > 0:
            weak_label = 1
        else:
            weak_label = 0

        out_dict = {}
        out_dict["y"] = weak_label
        if self.only_flair:
            im = ((im[0, :, :]).unsqueeze(0))

        # return im, out_dict, seg, x_path --> out_dict là {} --> ở đây tác giả ko dùng đến
        return im, weak_label, seg,sal, x_path 
        ## chỉ cần quan tâm tới đầu ra:
        ## image, label, seg (groundtruth_mask), saliency (được generate ra), path
    
## Dùng 6300 + rot90 để training DDPM
## Dùng dữ liệu training classifier để train classifier và Autoencoder
## 
'''