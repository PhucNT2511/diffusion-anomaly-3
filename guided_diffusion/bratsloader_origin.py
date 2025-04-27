import torch
import numpy as np
import pandas as pd

def normalize(image):
    """Chuẩn hóa dữ liệu theo min-max."""
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    if scale > 0:
        return (image - min_) / scale
    return image

def irm_min_max_preprocess(image, low_perc=1, high_perc=99):
    """Tiền xử lý dữ liệu bằng cách loại bỏ ngoại lệ và chuẩn hóa min-max."""
    non_zeros = image > 0
    if non_zeros.sum() > 0:
        low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
        image = np.clip(image, low, high)
        image = normalize(image)
    return image

class BRATSDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mode="train",
        fold=1,
        test_flag=False,
        transforms=None,
        few_shot="both_positive_negative",
    ):
        super().__init__()
        self.transforms = transforms
        self.few_shot = few_shot
        print("Sử dụng biến đổi dữ liệu cho tăng cường dữ liệu." if transforms else "Không sử dụng tăng cường dữ liệu.")

        # Tải danh sách đường dẫn dữ liệu và metadata
        data_split = np.load(
            '/kaggle/working/diffusion-anomaly-3/data/brats/data_split.npz',
            allow_pickle=True
        )
        meta_df = pd.read_csv(
            '/kaggle/working/diffusion-anomaly-3/data/brats/meta_data.csv'
        )
        volume_ids = data_split[f'{mode}_folds'].item()[f'fold_{fold}']
        if not test_flag:
            self.datapaths = meta_df[
                meta_df['volume'].isin(volume_ids)
            ]['path'].values
        else:
            self.datapaths = meta_df[
                (meta_df['volume'].isin(volume_ids)) & (meta_df['label'] == 1)
            ]['path'].values
        print(f'Số lượng dữ liệu {mode}: {len(self.datapaths)}')

        # Xác định ảnh few-shot và nhãn tồn tại annotation
        few_shot_dict = np.load(
            '/kaggle/working/diffusion-anomaly-3/data/brats/few_shot_path.npy',
            allow_pickle=True
        ).item()
        self.proto_paths = few_shot_dict[self.few_shot]
        self.exist_annotation = np.array([
            1 if p in self.proto_paths else 0 for p in self.datapaths
        ])

        # Tính nhãn cụm dựa trên khoảng cách tới các proto centers
        self.cluster_labels = self.perform_clustering()

    def perform_clustering(self):
        """
        Gán nhãn cụm cho mỗi ảnh trong dataset dựa trên ảnh few-shot prototypes.
        - Tải và tiền xử lý ảnh proto
        - Lấy vector đặc trưng bằng flatten
        - (Tùy chọn) Giảm chiều PCA
        - Tính khoảng cách Euclid tới mỗi proto và gán nhãn argmin
        Trả về:
        - cluster_labels: numpy array chứa nhãn cụm của từng ảnh
        """
        # Đọc và tiền xử lý proto centers
        centers = []
        for p in self.proto_paths:
            d = np.load(p)
            img = d['image'][[1,2,3,0],:,:]
            proc = np.array([irm_min_max_preprocess(img[i]) for i in range(img.shape[0])])
            centers.append(proc.flatten())
        centers = np.stack(centers)  # (n_centers, D)

        # Đọc và tiền xử lý toàn bộ dataset
        feats = []
        for p in self.datapaths:
            d = np.load(p)
            img = d['image'][[1,2,3,0],:,:]
            proc = np.array([irm_min_max_preprocess(img[i]) for i in range(img.shape[0])])
            feats.append(proc.flatten())
        feats = np.stack(feats)  # (n_samples, D)

        # Tính khoảng cách Euclid và gán nhãn
        dists = np.linalg.norm(
            feats[:, None, :] - centers[None, :, :],
            axis=2
        )
        labels = np.argmin(dists, axis=1)

        # Lưu kết quả ra CSV
        df = pd.DataFrame({
            'filepath': self.datapaths,
            'cluster_label': labels
        })
        df.to_csv('cluster_labels.csv', index=False)
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

'''
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
        self.cluster_labels = self.perform_clustering()

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
        d = np.load(self.datapaths[idx])
        img = d['image'][[1,2,3,0]]
        proc = np.stack([irm_min_max_preprocess(img[i]) for i in range(img.shape[0])])
        padding_img = np.zeros((4,256,256), dtype=np.float32)
        padding_img[:,8:-8,8:-8] = proc
        mask = d['mask']
        padding_mask = np.zeros((256,256), dtype=np.float32)
        padding_mask[8:-8,8:-8] = mask
        label = 1 if mask.sum() > 0 else 0
        img_t = torch.Tensor(padding_img)
        if self.transforms:
            img_t = self.transforms(img_t)
        return img_t, label, padding_mask, self.exist_annotation[idx], int(self.cluster_labels[idx])

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

# === Unified Forward & Backward ===
def forward_backward_log(
    w_loader, w_iter,
    a_loader, a_iter,
    c_loader, c_iter,
    model, diffusion, schedule_sampler, mp_trainer,
    lambda_0, lambda_1, lambda_2,
    noised=False
):
    """
    Fetch one batch from each loader, compute three losses (classification on weak batch with optional noise,
    annotation loss on annotated batch, and discrepancy+robustness on clustered batch), sum and backprop.
    Returns loss dict and updated iterators.
    """
    dev = next(model.parameters()).device

    # --- Classification Batch (weak loader) ---
    try:
        w_imgs, w_lbls, _, _, _ = next(w_iter)
    except StopIteration:
        w_iter = iter(w_loader)
        w_imgs, w_lbls, _, _, _ = next(w_iter)
    w_imgs, w_lbls = w_imgs.to(dev), w_lbls.to(dev)
    
    # Apply noise sampling if requested
    if noised and schedule_sampler is not None and diffusion is not None:
        t, _ = schedule_sampler.sample(w_imgs.shape[0], dev)
        w_noised = diffusion.q_sample(w_imgs, t)
    else:
        t = torch.zeros(w_imgs.shape[0], dtype=torch.long, device=dev)
        w_noised = w_imgs
    # Forward classification
    logits_w, _ = model(w_noised, w_imgs, timesteps=t)
    loss_bce = F.cross_entropy(logits_w, w_lbls)

    # --- Annotation Batch (only annotated) ---
    try:
        a_imgs, _, a_masks, _, _ = next(a_iter)
    except StopIteration:
        a_iter = iter(a_loader)
        a_imgs, _, a_masks, _, _ = next(a_iter)
    a_imgs, a_masks = a_imgs.to(dev), a_masks.to(dev)
    # Forward annotation
    logits_a, sal_a = model(a_imgs, a_imgs, timesteps=None)
    # Real annotation loss: only on annotated images
    real_idx = torch.arange(sal_a.size(0), device=dev)  # all in this loader are annotated
    loss_anno = F.binary_cross_entropy(sal_a[real_idx], a_masks.unsqueeze(1)[real_idx])

    # --- Clustered Batch (subclusters) ---
    try:
        c_imgs, _, _, c_ann, c_lbls = next(c_iter)
    except StopIteration:
        c_iter = iter(c_loader)
        c_imgs, _, _, c_ann, c_lbls = next(c_iter)
    c_imgs = c_imgs.to(dev)
    c_ann = c_ann.to(dev)
    c_lbls = c_lbls.to(dev)
    # Forward clustered
    logits_c, sal_c = model(c_imgs, c_imgs, timesteps=None)
    # Compute cluster centers (mean saliency per cluster)
    centers = {}
    for cl in c_lbls.unique():
        idxs = (c_lbls == cl).nonzero(as_tuple=True)[0]
        centers[cl.item()] = sal_c[idxs].mean(dim=0)
    # Discrepancy: between annotated and non-annotated
    disc_loss = torch.tensor(0., device=dev)
    for cl in centers:
        mask_real = (c_lbls == cl) & (c_ann == 1)
        mask_gen = (c_lbls == cl) & (c_ann == 0)
        if mask_real.any() and mask_gen.any():
            real_vals = sal_c[mask_real]
            gen_vals = sal_c[mask_gen]
            diffs = real_vals.unsqueeze(1) - gen_vals.unsqueeze(0)
            disc_loss += torch.sqrt(diffs.pow(2).mean(dim=(2,3))).sum() / (mask_real.sum() * mask_gen.sum())
    # Robustness: within non-annotated
    rob_loss = torch.tensor(0., device=dev)
    for cl, center in centers.items():
        mask_gen = (c_lbls == cl) & (c_ann == 0)
        if mask_gen.sum() > 1:
            gen_vals = sal_c[mask_gen]
            diffs = gen_vals.unsqueeze(1) - gen_vals.unsqueeze(0)
            rob_loss += torch.sqrt(diffs.pow(2).mean(dim=(2,3))).sum() / (mask_gen.sum()**2)
    # Normalize losses
    disc_loss = disc_loss / len(centers)
    rob_loss = rob_loss / len(centers)

    # --- Total Loss and Backprop ---
    total_loss = loss_bce + lambda_0 * loss_anno + lambda_1 * disc_loss + lambda_2 * rob_loss
    mp_trainer.zero_grad()
    total_loss.backward()
    mp_trainer.step()

    loss_dict = {
        'bce': loss_bce.item(),
        'anno': loss_anno.item(),
        'disc': disc_loss.item(),
        'rob': rob_loss.item(),
        'total': total_loss.item()
    }
    return loss_dict, (w_iter, a_iter, c_iter)

# === Training Cycle ===
def train_cycle(
    w_loader, a_loader, c_loader,
    w_iter, a_iter, c_iter,
    model, diffusion, schedule_sampler, mp_trainer,
    lambda_0, lambda_1, lambda_2, steps
):
    for step in range(steps):
        losses, (w_iter, a_iter, c_iter) = forward_backward_log(
            w_loader, w_iter, a_loader, a_iter, c_loader, c_iter,
            model, diffusion, schedule_sampler, mp_trainer,
            lambda_0, lambda_1, lambda_2
        )
        print(f"Step {step}: {losses}")
    return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--subclusters_per_batch", type=int, default=4)
    parser.add_argument("--min_cluster_size", type=int, default=3)
    parser.add_argument("--max_cluster_size", type=int, default=10)
    parser.add_argument("--lambda_0", type=float, default=1.0)
    parser.add_argument("--lambda_1", type=float, default=1.0)
    parser.add_argument("--lambda_2", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    ds = BRATSDataset()
    w_ds, a_ds, c_ds, subcs = split_dataset_by_annotation_and_cluster(
        ds, args.min_cluster_size, args.max_cluster_size
    )
    w_loader = DataLoader(w_ds, batch_size=args.batch_size, shuffle=True)
    a_loader = DataLoader(a_ds, batch_size=args.batch_size, shuffle=True)
    c_sampler = SubClusterBatchSampler(subcs, args.subclusters_per_batch)
    c_loader = DataLoader(c_ds, batch_sampler=c_sampler)
    w_iter, a_iter, c_iter = iter(w_loader), iter(a_loader), iter(c_loader)

    # initialize your model, diffusion, schedule_sampler, mp_trainer here
    model = ...
    diffusion = ...
    schedule_sampler = ...
    mp_trainer = ...

    train_cycle(
        w_loader, a_loader, c_loader,
        w_iter, a_iter, c_iter,
        model, diffusion, schedule_sampler, mp_trainer,
        args.lambda_0, args.lambda_1, args.lambda_2, args.steps
    )


'''