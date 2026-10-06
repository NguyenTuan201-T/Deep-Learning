


import os, shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from google.colab import drive

drive.mount('/content/drive')


DATA_DIR = '/content/drive/MyDrive/CXR_Project/data/'          # có images_224.npy, meta.csv, val.csv, test.csv
BEST_CKPT = '/content/drive/MyDrive/CXR_Project/runs/densenet121_baseline/best.pt'   #  đường dẫn best.pt từ repo
OUT_DIR = '/content/preds/'                                    # nơi lưu 4 file đầu ra

DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia',
]
EXPECTED = {'val': (10664, 14), 'test': (25596, 14)}
BATCH_SIZE, NUM_WORKERS = 32, 2
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = device.type == 'cuda'
print('Device:', device)

# ---------- 1. Ảnh----------
src = DATA_DIR + 'images_224.npy'
IMAGES_NPY = '/content/images_224.npy'
assert os.path.exists(src), f'Không thấy {src}'
if not (os.path.exists(IMAGES_NPY) and os.path.getsize(IMAGES_NPY) == os.path.getsize(src)):
    print('Đang copy images_224.npy về đĩa Colab (chỉ lần đầu)...')
    shutil.copyfile(src, IMAGES_NPY)
imgs = np.load(IMAGES_NPY, mmap_mode='r')
SCALE = 255.0 if (imgs.dtype == np.uint8 or float(np.asarray(imgs[:64]).max()) > 1.5) else 1.0
print('Mảng ảnh:', imgs.shape, imgs.dtype, '| thang chia:', SCALE)

# ---------- 2. CSV + cột idx ----------
meta = pd.read_csv(DATA_DIR + 'meta.csv')
assert len(meta) == len(imgs), f'meta.csv {len(meta)} dòng, mảng ảnh {len(imgs)} ảnh'
pos = {name: i for i, name in enumerate(meta['image'])}

def load_csv(name):
    d = pd.read_csv(DATA_DIR + f'{name}.csv')
    if 'idx' not in d.columns:                      # idx = vị trí ảnh trong images_224.npy
        d['idx'] = d['image'].map(pos)
        assert d['idx'].notna().all(), f'{name}.csv có ảnh không có trong meta.csv'
        d['idx'] = d['idx'].astype(int)
    assert d['idx'].min() >= 0 and d['idx'].max() < len(imgs), f'idx của {name} vượt phạm vi'
    return d

val_df, test_df = load_csv('val'), load_csv('test')
print(f'val: {len(val_df)} ảnh | test: {len(test_df)} ảnh')

# 3. Dataset / loader / tiền xử lý 
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

class CXRDataset(Dataset):
    def __init__(self, df):
        self.idx = df['idx'].to_numpy(dtype=np.int64)
        self.y = df[DISEASES].to_numpy(dtype=np.float32)
        self.arr = None                          
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        if self.arr is None:
            self.arr = np.load(IMAGES_NPY, mmap_mode='r')
        a = np.array(self.arr[self.idx[i]])
        if a.ndim == 2:
            a = a[None]                              # (H,W) -> (1,H,W)
        elif a.shape[-1] in (1, 3):
            a = a.transpose(2, 0, 1)                 # (H,W,C) -> (C,H,W)
        return torch.from_numpy(np.ascontiguousarray(a)), torch.from_numpy(self.y[i])

def make_loader(df):
    return DataLoader(CXRDataset(df), batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)

def prep(x):
    x = x.to(device, non_blocking=True).float() / SCALE
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    return (x - MEAN) / STD

@torch.no_grad()
def predict(model, loader):
    model.eval()
    out_l, out_y = [], []
    for x, y in tqdm(loader, leave=False):
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            logits = model(prep(x))
        out_l.append(logits.float().cpu().numpy())
        out_y.append(y.numpy())
    return np.concatenate(out_l), np.concatenate(out_y)

def macro_auroc(logits, labels):
    probs = 1 / (1 + np.exp(-logits))                # sigmoid: logit -> xác suất
    aucs = [np.nan if labels[:, c].min() == labels[:, c].max()
            else roc_auc_score(labels[:, c], probs[:, c]) for c in range(labels.shape[1])]
    return float(np.nanmean(aucs)), aucs

# 4. Nạp best.pt ----------
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, len(DISEASES)) 
ck = torch.load(BEST_CKPT, map_location=device)
model.load_state_dict(ck['model'])
model.to(device)
print(f"Đã nạp best.pt: epoch {ck['epoch'] + 1}, val macro AUROC lúc lưu = {ck['val_auroc']:.4f}")

#  5. Dự đoán val và test 
v_logits, v_labels = predict(model, make_loader(val_df))
t_logits, t_labels = predict(model, make_loader(test_df))

# 6. Kiểm tra trước khi lưu 
for split, lg, lb, df in [('val', v_logits, v_labels, val_df), ('test', t_logits, t_labels, test_df)]:
    assert lg.shape == lb.shape == EXPECTED[split], f'{split}: {lg.shape} / {lb.shape}, cần {EXPECTED[split]}'
    assert np.isfinite(lg).all(), f'{split}: logits có NaN hoặc vô cực'
    assert np.array_equal(lb, df[DISEASES].to_numpy(dtype=np.float32)), f'{split}: nhãn lệch thứ tự so với CSV'
    print(f'{split}: shape {lg.shape} OK')

auc_val, auc_each = macro_auroc(v_logits, v_labels)
print(f"val macro AUROC tính lại = {auc_val:.4f} (lúc lưu: {ck['val_auroc']:.4f})")
assert abs(auc_val - ck['val_auroc']) < 1e-3, 'Không khớp với lúc lưu best.pt: nạp nhầm checkpoint hoặc sai tiền xử lý'

#7. Lưu 4 file (logits chưa qua sigmoid) và đọc lại để xác nhận
for name, arr in [('logits_val', v_logits), ('labels_val', v_labels),
                  ('logits_test', t_logits), ('labels_test', t_labels)]:
    arr = arr.astype(np.float32)
    np.save(OUT_DIR + name + '.npy', arr)
    back = np.load(OUT_DIR + name + '.npy')
    assert back.shape == arr.shape and np.array_equal(back, arr), name
    print(f'{name}.npy: {back.shape}, {back.dtype}')

print('AUROC val từng bệnh:')
print(pd.Series(auc_each, index=DISEASES).round(4).to_string())

#8. (Tùy chọn) Xác suất từ logit, dùng cho threshold / calibration ----------
probs_val = 1 / (1 + np.exp(-v_logits))              # (10664, 14), mỗi giá trị trong [0, 1]
pred_val = probs_val >= 0.5                          # dự đoán nhị phân theo ngưỡng 0.5
print('Xác suất val: min %.4f, max %.4f' % (probs_val.min(), probs_val.max()))

