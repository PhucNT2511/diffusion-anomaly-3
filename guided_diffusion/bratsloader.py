import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset, Subset, DataLoader, Sampler
import torch.nn.functional as F

# === Preprocessing ===
def normalize(image):
    min_, max_ = image.min(), image.max()
    return (image - min_) / (max_ - min_) if max_ > min_ else image

def irm_min_max_preprocess(image, low_perc=1, high_perc=99):
    non_zero = image > 0
    if non_zero.sum() > 0:
        low, high = np.percentile(image[non_zero], [low_perc, high_perc])
        image = np.clip(image, low, high)
        image = normalize(image)
    return image

# === Dataset & Clustering ===
class BRATSDataset(Dataset):
    def __init__(self, mode="train", fold=1, test_flag=False, transforms=None, few_shot="both_positive_negative"):
        super().__init__()
        self.transforms = transforms
        # load split and metadata
        split = np.load('/kaggle/working/diffusion-anomaly-3/data/brats/data_split.npz', allow_pickle=True)
        meta = pd.read_csv('/kaggle/working/diffusion-anomaly-3/data/brats/meta_data.csv')
        vols = split[f'{mode}_folds'].item()[f'fold_{fold}']
        df = meta[meta['volume'].isin(vols)]
        if test_flag:
            df = df[df['label']==1]
        self.datapaths = df['path'].values
        # annotation flags
        few_shot = np.load('/kaggle/working/diffusion-anomaly-3/data/brats/few_shot_path.npy', allow_pickle=True).item()
        self.proto_paths = few_shot[few_shot]
        self.exist_annotation = np.array([1 if p in self.proto_paths else 0 for p in self.datapaths])
        
        # cluster labels
        #self.cluster_labels = self.perform_clustering()
        self.cluster_labels = pd.read_csv(
            f'data/brats/{self.mode}_cluster_labels.csv'
        )['cluster_label']


    def perform_clustering(self):
        # compute centers
        centers = []
        for p in self.proto_paths:
            d = np.load(p)
            img = d['image'][[1,2,3,0]]
            proc = np.stack([irm_min_max_preprocess(img[i]) for i in range(img.shape[0])]).flatten()
            centers.append(proc)
        centers = np.stack(centers)
        # compute features
        feats = []
        for p in self.datapaths:
            d = np.load(p)
            img = d['image'][[1,2,3,0]]
            proc = np.stack([irm_min_max_preprocess(img[i]) for i in range(img.shape[0])]).flatten()
            feats.append(proc)
        feats = np.stack(feats)
        # assign labels
        dists = np.linalg.norm(feats[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        # save
        pd.DataFrame({'filepath': self.datapaths, 'cluster_label': labels}).to_csv('cluster_labels.csv', index=False)
        return labels

    def __getitem__(self, idx):
        # Tải và tiền xử lý
        d = np.load(self.datapaths[idx])
        img = d['image'][[1,2,3,0],:,:]
        for i in range(img.shape[0]):
            img[i] = irm_min_max_preprocess(img[i])

        mask = d['mask']
        # padding
        padding_img = np.zeros((4,256,256), dtype=np.float32)
        padding_img[:,8:-8,8:-8] = img
        padding_mask = np.zeros((256,256), dtype=np.float32)
        padding_mask[8:-8,8:-8] = mask

        label = 1 if mask.sum() > 0 else 0
        cond = {'y': label}
        if self.transforms:
            padding_img = self.transforms(torch.Tensor(padding_img))

        return (
            padding_img,
            cond,
            label,
            padding_mask,
            self.exist_annotation[idx],
            int(self.cluster_labels[idx])
        )


    def __len__(self):
        return len(self.datapaths)

# === Split & Sub-clusters with annotations included ===
def split_dataset_by_annotation_and_cluster(dataset, min_cluster_size=3, max_cluster_size=10):
    # weak and annotated subsets
    weak_ds = Subset(dataset, list(range(len(dataset))))
    ann_ds = Subset(dataset, [i for i, f in enumerate(dataset.exist_annotation) if f])
    # group by cluster
    cluster_to_inds = {}
    for i, c in enumerate(dataset.cluster_labels):
        cluster_to_inds.setdefault(c, []).append(i)
    # build subclusters
    subclusters = []
    for inds in cluster_to_inds.values():
        ann = [i for i in inds if dataset.exist_annotation[i] == 1]
        non_ann = [i for i in inds if dataset.exist_annotation[i] == 0]
        random.shuffle(non_ann)
        # create subclusters ensuring at least one annotated per cluster when possible
        idx_na = 0
        while idx_na < len(non_ann):
            size = random.randint(min_cluster_size, min(max_cluster_size, len(non_ann) - idx_na + (1 if ann else 0)))
            sub = []
            if ann:
                sub.append(ann.pop())
            take = min(size - len(sub), len(non_ann) - idx_na)
            sub.extend(non_ann[idx_na:idx_na+take])
            idx_na += take
            subclusters.append(sub)
        # leftover annotated-only
        for a in ann:
            subclusters.append([a])
    # clustered dataset
    all_idx = list({i for sub in subclusters for i in sub})
    clus_ds = Subset(dataset, all_idx)
    return weak_ds, ann_ds, clus_ds, subclusters

class SubClusterBatchSampler(Sampler):
    def __init__(self, subclusters, batch_size):
        self.subclusters = subclusters
        self.batch_size = batch_size
    def __iter__(self):
        random.shuffle(self.subclusters)
        for i in range(0, len(self.subclusters), self.batch_size):
            yield self.subclusters[i:i+self.batch_size]
    def __len__(self):
        return (len(self.subclusters) + self.batch_size - 1) // self.batch_size