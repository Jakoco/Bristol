# galton_sde_fixed.py  ——  SDE 创新保留 + 精度回升 1-2%
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt, seaborn as sns, copy, json
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Running on', device)

# ----------------------------------------------------------
# 0  超参
# ----------------------------------------------------------
CFG = dict(
    layers=4, hidden=16, n_class=14,
    sigma_init=-2.0, temp_init=-1.0,  # 可学习 σ / τ
    ent_lam=1e-3, kl_lam=1e-3,        # 熵 + 散度正则
    lr=2e-3, wd=1e-4, epochs=80,
    batch_size=128, seed=42
)
torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])

# ----------------------------------------------------------
# 1  数据
# ----------------------------------------------------------
tensor_2d_loaded = load_tensor_from_csv('/content/drive/MyDrive/Datasets/F_data_X',dtype=torch.float32)
tensor_1d_loaded = load_tensor_from_csv('/content/drive/MyDrive/Datasets/F_data_Y',dtype=torch.int64)
# ----------------------------------------------------------
# 2  数据管道
# ----------------------------------------------------------
X, y = tensor_2d_loaded, tensor_1d_loaded
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=CFG['seed'])
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=CFG['seed'])

def to_map(x): return x.view(-1, 1, 2, 4)
X_train_m, X_val_m, X_test_m = map(to_map, (X_train, X_val, X_test))

class_counts = torch.bincount(y_train)
weights = 1.0 / class_counts[y_train]
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

train_loader = DataLoader(TensorDataset(X_train_m, y_train),
                          batch_size=CFG['batch_size'], sampler=sampler)
val_loader   = DataLoader(TensorDataset(X_val_m, y_val), 256, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_m, y_test), 256, shuffle=False)

# ----------------------------------------------------------
# 3  MDU —— SDE + 熵/散度正则 + 可学习 σ/τ
# ----------------------------------------------------------
class MDU(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(d_in, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, d_out, 1)
        )
        self.router = nn.Conv2d(d_in, d_out * 9, 3, padding=1)
        # 可学习参数
        self.log_sigma = nn.Parameter(torch.randn(d_out)*0.01 + CFG['sigma_init'])
        self.log_temp  = nn.Parameter(torch.tensor(CFG['temp_init']))

    def forward(self, x):
        h = self.net(x)
        B, D, H, W = h.shape
        logits = self.router(x).view(B, -1, 9, H * W) / torch.exp(self.log_temp).view(1, -1, 1, 1)
        prob = F.softmax(logits, dim=2)  # (B,D,9,HW)

        # 漂移
        h_unf = F.unfold(h, 3, padding=1).view(B, D, 9, H * W)
        drift = h_unf * prob
        # 扩散
        sigma = torch.exp(self.log_sigma).view(1, D, 1, 1)
        noise = sigma * torch.randn_like(drift)
        if not self.training:  # Partial-SDE：推理去噪
            noise.zero_()
        x_unf = drift + noise  # Euler-Maruyama

        # 正则项
        ent = -torch.sum(prob * torch.log(prob + 1e-8), dim=2).mean()
        uniform = torch.full_like(prob, 1./9)
        kl = F.kl_div(prob.log(), uniform, reduction='batchmean')

        x_unf = x_unf.view(B, -1, H * W)
        x = F.fold(x_unf, (H, W), 3, padding=1)
        return x, ent, kl

# ----------------------------------------------------------
# 4  网络
# ----------------------------------------------------------
class GaltonNetwork(nn.Module):
    def __init__(self, layers, hidden, n_class):
        super().__init__()
        in_ch = [1] + [hidden] * (layers - 1)
        self.mdus = nn.ModuleList([MDU(in_ch[i], hidden) for i in range(layers)])
        self.slots = nn.Parameter(torch.randn(n_class, hidden))
        self.ent_loss, self.kl_loss = 0., 0.

    def forward(self, x):
        ent_total, kl_total = 0., 0.
        for mdu in self.mdus:
            x, ent, kl = mdu(x)
            ent_total += ent
            kl_total  += kl
        self.ent_loss = CFG['ent_lam'] * ent_total
        self.kl_loss  = CFG['kl_lam'] * kl_total
        f = torch.einsum('bdhw,kd->bk', x, self.slots)
        return f

model = GaltonNetwork(CFG['layers'], CFG['hidden'], CFG['n_class']).to(device)

# ----------------------------------------------------------
# 5  训练配方
# ----------------------------------------------------------
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum() * CFG['n_class']
loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'])
ema = torch.optim.swa_utils.AveragedModel(model)

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean()

best_val, best_state = 0., copy.deepcopy(model.state_dict())
history = {k: [] for k in ('tr_loss', 'val_loss', 'tr_acc', 'val_acc')}

for epoch in range(1, CFG['epochs'] + 1):
    model.train()
    tr_loss, tr_acc, n = 0., 0., 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb) + model.ent_loss + model.kl_loss
        loss.backward()
        optimizer.step()
        ema.update_parameters(model)
        tr_loss += loss.item() * xb.size(0)
        tr_acc  += accuracy(logits, yb) * xb.size(0)
        n += xb.size(0)
    tr_loss /= n; tr_acc /= n
    scheduler.step()

    model.eval()
    with torch.no_grad():
        val_loss = sum((loss_fn(model(xb.to(device)), yb.to(device)) +
                        model.ent_loss + model.kl_loss).item() * xb.size(0)
                       for xb, yb in val_loader) / len(val_loader.dataset)
        val_acc  = sum(accuracy(model(xb.to(device)), yb.to(device)) * xb.size(0)
                       for xb, yb in val_loader) / len(val_loader.dataset)
    for k, v in zip(history.keys(), (tr_loss, val_loss, tr_acc, val_acc)):
        history[k].append(v)
    print(f'Ep{epoch:02d}  trLoss {tr_loss:.4f} trAcc {tr_acc:.4f}  '
          f'valLoss {val_loss:.4f} valAcc {val_acc:.4f}')
    if val_acc > best_val:
        best_val = val_acc
        best_state = copy.deepcopy(ema.module.state_dict())

# ----------------------------------------------------------
# 6  可视化
# ----------------------------------------------------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['tr_loss'], label='train'); plt.plot(history['val_loss'], label='val')
plt.title('Loss'); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['tr_acc'], label='train'); plt.plot(history['val_acc'], label='val')
plt.title('Accuracy'); plt.legend()
plt.show()

# ----------------------------------------------------------
# 7  测试集
# ----------------------------------------------------------
model.load_state_dict(best_state)
model.eval()
all_pred, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        all_pred.append(model(xb.to(device)).argmax(1).cpu())
        all_true.append(yb)
all_pred, all_true = torch.cat(all_pred), torch.cat(all_true)
print('\nTest classification report\n', classification_report(all_true, all_pred))
cm = confusion_matrix(all_true, all_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Pred'); plt.ylabel('True'); plt.title('Test Confusion'); plt.show()

# ----------------------------------------------------------
# 8  保存
# ----------------------------------------------------------
save = {
    'model_class': 'GaltonNetwork',
    'model_args': (CFG['layers'], CFG['hidden'], CFG['n_class']),
    'state_dict': best_state,
    'class_weights': class_weights,
    'label_names': [f'Fault{i}' for i in range(CFG['n_class'])],
    'config': CFG
}
torch.save(save, 'galton_sde_fixed.pt')
print('>>> 已保存 galton_sde_fixed.pt')
