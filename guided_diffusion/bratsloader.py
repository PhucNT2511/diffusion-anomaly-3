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
