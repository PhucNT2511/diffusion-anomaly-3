import torch 
import numpy as np
import torch.nn.functional as F
import pickle
import pandas as pd

def normalize(image):
    """Basic min max scaler.
    """
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
    return image

'''
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
'''

class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, mode="train", fold=1, test_flag=False, transforms=None):
        
        super().__init__()
        self.datapaths = []
        self.transforms = transforms
        if self.transforms:
            print("Transform for data augmentation.")
        else:
            print("No data augmentation")
        

        meta_data_df = pd.read_csv('/kaggle/working/diffusion-anomaly-3/data/brats/total_authentic_and_synthetic.csv')
        self.datapaths = meta_data_df['path'].values
        self.labels = meta_data_df['label'].values
        print(f'Number of {mode} data: {len(self.datapaths)}')

    def __getitem__(self, idx):
        data = np.load(self.datapaths[idx])
        if self.labels[idx] == 0:
            image = data['image']
        else:
            image = data
        image = image[[1, 2, 3, 0], :, :]
        for i in range(image.shape[0]):
            image[i] = normalize(image[i])
        
        padding_image = np.zeros((4, 256, 256))
        padding_image[:, 8:-8, 8:-8] = image
        
        cond = {}
        cond['y'] = self.label
        if self.transforms:
            padding_image = self.transforms(torch.Tensor(padding_image))

        return np.float32(padding_image), cond, self.label

    def __len__(self):
        return len(self.datapaths)
