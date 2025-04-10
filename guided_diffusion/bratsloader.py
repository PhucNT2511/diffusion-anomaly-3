import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import faiss

def normalize(image):
    """Chuẩn hóa dữ liệu theo min-max."""
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
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
    def __init__(self, mode="train", fold=1, test_flag=False, transforms=None, few_shot="both_positive_negative"):
        super().__init__()
        self.datapaths = []
        self.transforms = transforms
        if self.transforms:
            print("Sử dụng biến đổi dữ liệu cho tăng cường dữ liệu.")
        else:
            print("Không sử dụng tăng cường dữ liệu.")

        # Tải danh sách đường dẫn dữ liệu và thông tin liên quan
        data_split = np.load('/kaggle/working/diffusion-anomaly-3/data/brats/data_split.npz', allow_pickle=True)
        meta_data_df = pd.read_csv('/kaggle/working/diffusion-anomaly-3/data/brats/meta_data.csv')
        volume_ids = data_split[f'{mode}_folds'].item()[f'fold_{fold}']
        if not test_flag:
            self.datapaths = meta_data_df[meta_data_df['volume'].isin(volume_ids)]['path'].values
        else:
            self.datapaths = meta_data_df[(meta_data_df['volume'].isin(volume_ids)) & (meta_data_df['label'] == 1)]['path'].values
        print(f'Số lượng dữ liệu {mode}: {len(self.datapaths)}')

        # Xác định các ảnh có annotation tồn tại
        self.exist_annotation = None
        if few_shot in ['only_positive', 'both_positive_negative']:
            few_shot_path = np.load('/kaggle/working/diffusion-anomaly-3/data/brats/few_shot_path.npy', allow_pickle=True).item()[few_shot]
            self.exist_annotation = np.array([1 if path in few_shot_path else 0 for path in self.datapaths])

        # Tiền xử lý và phân cụm dữ liệu
        self.cluster_labels = self.perform_clustering(n_clusters=4, n_iter=300, normalize=True)

    def perform_clustering(self, n_clusters=4, n_iter=300, normalize=True):
        """
        Tiền xử lý dữ liệu và thực hiện phân cụm sử dụng MiniBatchKMeans sau khi giảm chiều bằng PCA.

        Trả về:
        - cluster_labels: numpy array chứa nhãn cụm của từng ảnh.
        """
        # Đọc và tiền xử lý dữ liệu
        dataset = []
        for path in self.datapaths:
            data = np.load(path)
            image = data['image']
            image = image[[1, 2, 3, 0], :, :]  # Sắp xếp lại các kênh
            processed_image = np.array([irm_min_max_preprocess(image[i]) for i in range(image.shape[0])])
            dataset.append(processed_image)
        dataset = np.array(dataset)

        # Reshape dữ liệu thành (N, 256*256*4)
        N = dataset.shape[0]
        X = dataset.reshape(N, -1).astype('float32')

        # Giảm chiều dữ liệu bằng PCA
        pca = PCA(n_components=50)  # Số lượng thành phần chính giữ lại, có thể điều chỉnh
        X_reduced = pca.fit_transform(X)

        # Chuẩn hóa dữ liệu nếu cần
        if normalize:
            norms = np.linalg.norm(X_reduced, axis=1, keepdims=True)
            X_reduced = X_reduced / (norms + 1e-10)

        # Thực hiện phân cụm bằng MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init='auto', max_iter=n_iter, batch_size=100)
        cluster_labels = kmeans.fit_predict(X_reduced)

        return cluster_labels

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
        cond = {'y': label}
        if self.transforms:
            padding_image = self.transforms(torch.Tensor(padding_image))

        return (np.float32(padding_image), cond, label, np.float32(padding_mask),
                self.exist_annotation[idx], self.cluster_labels[idx])

    def __len__(self):
        return len(self.datapaths)
