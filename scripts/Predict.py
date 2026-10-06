"""
predict_export.py - Người 2: nạp best.pt, chạy trên val và test, lưu 4 file .npy
và kiểm tra kích thước (10664, 14) / (25596, 14).

Dùng đúng pipeline của Khối C (DenseNet-121, images_224.npy + cột idx,
chuẩn hóa ImageNet, ảnh xám nhân ra 3 kênh), nên kết quả tái lập được.

Chạy trên Colab:
    from google.colab import drive; drive.mount('/content/drive')
    !python predict_export.py \
        --data_dir /content/drive/MyDrive/CXR_Project/data/ \
        --ckpt best.pt \
        --out_dir preds

Đầu ra trong --out_dir:
    logits_val.npy   (10664, 14) float32   logits CHƯA qua sigmoid
    logits_test.npy  (25596, 14) float32
    labels_val.npy   (10664, 14) float32   0/1
    labels_test.npy  (25596, 14) float32
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm.auto import tqdm

DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia',
]
EXPECTED = {'val': (10664, 14), 'test': (25596, 14)}
BATCH_SIZE = 32
NUM_WORKERS = 2


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True,
                    help='thư mục có images_224.npy, meta.csv, val.csv, test.csv')
    ap.add_argument('--ckpt', default='best.pt', help='đường dẫn best.pt')
    ap.add_argument('--out_dir', default='preds')
    ap.add_argument('--copy_to_local', action='store_true',
                    help='copy images_224.npy về đĩa Colab trước (nhanh hơn nhiều so với đọc từ Drive)')
    return ap.parse_args()


class CXRDataset(Dataset):
    """Lấy ảnh theo cột idx của dòng CSV; thứ tự mẫu = thứ tự dòng CSV."""

    def __init__(self, df, images_npy):
        self.images_npy = images_npy
        self.idx = df['idx'].to_numpy(dtype=np.int64)
        self.y = df[DISEASES].to_numpy(dtype=np.float32)
        self.arr = None  # mở mmap lười trong từng worker

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self.arr is None:
            self.arr = np.load(self.images_npy, mmap_mode='r')
        a = np.array(self.arr[self.idx[i]])
        if a.ndim == 2:                      # (H, W) -> (1, H, W)
            a = a[None]
        elif a.shape[-1] in (1, 3):          # (H, W, C) -> (C, H, W)
            a = a.transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(a)), torch.from_numpy(self.y[i])


def add_idx(df, meta, n_imgs, name):
    """CSV của Khối B không có cột idx: idx = vị trí ảnh trong images_224.npy
    (npy xếp ảnh theo đúng thứ tự dòng của meta.csv)."""
    if 'idx' in df.columns:
        return df
    pos = {img: i for i, img in enumerate(meta['image'])}
    df = df.copy()
    df['idx'] = df['image'].map(pos)
    assert df['idx'].notna().all(), f'{name}.csv có ảnh không có trong meta.csv'
    df['idx'] = df['idx'].astype(int)
    assert df['idx'].min() >= 0 and df['idx'].max() < n_imgs, f'idx của {name} vượt phạm vi'
    return df


@torch.no_grad()
def predict(model, loader, prep, device, use_amp):
    model.eval()
    out_l, out_y = [], []
    for x, y in tqdm(loader, leave=False):
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            logits = model(prep(x))
        out_l.append(logits.float().cpu().numpy())
        out_y.append(y.numpy())
    return np.concatenate(out_l), np.concatenate(out_y)


def macro_auroc(logits, labels):
    probs = 1 / (1 + np.exp(-logits))
    aucs = []
    for c in range(labels.shape[1]):
        if labels[:, c].min() == labels[:, c].max():
            aucs.append(np.nan)
        else:
            aucs.append(roc_auc_score(labels[:, c], probs[:, c]))
    return float(np.nanmean(aucs)), aucs


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print('Device:', device)

    # --- Ảnh ---
    images_npy = os.path.join(args.data_dir, 'images_224.npy')
    assert os.path.exists(images_npy), f'Không thấy {images_npy}'
    if args.copy_to_local:
        local = '/content/images_224.npy'
        if not (os.path.exists(local) and os.path.getsize(local) == os.path.getsize(images_npy)):
            print('Đang copy images_224.npy về đĩa local (chỉ lần đầu)...')
            shutil.copyfile(images_npy, local)
        images_npy = local
    imgs = np.load(images_npy, mmap_mode='r')
    scale = 255.0 if (imgs.dtype == np.uint8 or float(np.asarray(imgs[:64]).max()) > 1.5) else 1.0
    print('Mảng ảnh:', imgs.shape, imgs.dtype, '| thang chia:', scale)

    # --- CSV ---
    meta = pd.read_csv(os.path.join(args.data_dir, 'meta.csv'))
    assert len(meta) == len(imgs), f'meta.csv {len(meta)} dòng, mảng ảnh {len(imgs)} ảnh'
    val_df = add_idx(pd.read_csv(os.path.join(args.data_dir, 'val.csv')), meta, len(imgs), 'val')
    test_df = add_idx(pd.read_csv(os.path.join(args.data_dir, 'test.csv')), meta, len(imgs), 'test')
    print(f'val: {len(val_df)} ảnh | test: {len(test_df)} ảnh')

    # --- Tiền xử lý (giống hệt lúc train) ---
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def prep(x):
        x = x.to(device, non_blocking=True).float() / scale
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        return (x - mean) / std

    def make_loader(df):
        return DataLoader(CXRDataset(df, images_npy), batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    # --- Mô hình: nạp best.pt ---
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(DISEASES))
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck['model'])
    model.to(device)
    print(f"Đã nạp {args.ckpt}: epoch {ck['epoch'] + 1}, val macro AUROC lúc lưu = {ck['val_auroc']:.4f}")

    # --- Dự đoán ---
    v_logits, v_labels = predict(model, make_loader(val_df), prep, device, use_amp)
    t_logits, t_labels = predict(model, make_loader(test_df), prep, device, use_amp)

    # --- Kiểm tra ---
    for split, lg, lb, df in [('val', v_logits, v_labels, val_df), ('test', t_logits, t_labels, test_df)]:
        assert lg.shape == lb.shape == EXPECTED[split], f'{split}: {lg.shape} / {lb.shape}, cần {EXPECTED[split]}'
        assert np.isfinite(lg).all(), f'{split}: logits có NaN hoặc vô cực'
        assert np.array_equal(lb, df[DISEASES].to_numpy(dtype=np.float32)), f'{split}: nhãn lệch thứ tự so với CSV'
        print(f'{split}: shape {lg.shape} OK')

    auc_val, _ = macro_auroc(v_logits, v_labels)
    print(f"val macro AUROC tính lại = {auc_val:.4f} (lúc lưu: {ck['val_auroc']:.4f})")
    assert abs(auc_val - ck['val_auroc']) < 1e-3, 'Không khớp với lúc lưu best.pt: nạp nhầm checkpoint hoặc sai tiền xử lý'

    # --- Lưu 4 file ---
    os.makedirs(args.out_dir, exist_ok=True)
    for name, arr in [('logits_val', v_logits), ('labels_val', v_labels),
                      ('logits_test', t_logits), ('labels_test', t_labels)]:
        path = os.path.join(args.out_dir, name + '.npy')
        np.save(path, arr.astype(np.float32))
        back = np.load(path)
        assert back.shape == arr.shape and np.array_equal(back, arr.astype(np.float32)), name
        print(f'Đã lưu {path}: {back.shape}, {back.dtype}')


if __name__ == '__main__':
    main()