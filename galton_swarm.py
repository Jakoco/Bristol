# galton_swarm.py —— 高尔顿板群智网络（Galton Swarm Network）
#
# 旧 MDU 审计的三条失败 → 本原型的三条对策：
#   1) "全局 SDE 替代梯度下降" → 训练用 (μ,λ)-NES 进化策略，全文无 loss.backward()；
#      前向是真正的多步随机过程（sde 模式 = 钉板二项发散的扩散极限，Euler–Maruyama；
#      bean 模式 = 离散豆子），训练/推理全程采样，绝不去噪（对比旧 MDU 的 noise.zero_()）。
#   2) "读出不是高尔顿的" → 终点位置做 soft 直方图，类别分布 = 粒子计数归一化，
#      没有任何线性投影头（没有 slots / einsum）。
#   3) "数据流不可读、交互被堵" → 粒子是持久实体（P 个 bean 落 T 排钉，轨迹可追踪）；
#      群密度 ρ 逐步回授钉子偏置（拥挤侧向力）——交互是真实的介质耦合；
#      probe() 输出轨迹 / 密度场 / 逐时证据流，"信息到底传了什么"可以直接看。
#
# 运行：Colab GPU 上 %run galton_swarm.py（合成数据，约 1–2 分钟）；
#       换真实数据见第 1 节注释。本地 CPU 仅够冒烟（把 gens 调到 5–10 即可）。

import torch, torch.nn as nn
import numpy as np, matplotlib.pyplot as plt, time, math

# ----------------------------------------------------------
# 0  超参
# ----------------------------------------------------------
CFG = dict(
    F=8, K=14, N=2100,            # 特征数 / 类别数 / 样本数（合成数据）
    P=256,                        # 粒子数 = 隐式 ensemble 的大小
    T=24,                         # 钉板排数 = SDE 步数
    G=33,                         # 每排钉子列数（板宽 ±(G-1)/2·Δ）
    M=16,                         # 上下文维度：φ = W_enc·x 调制钉板
    dt=1.0, Delta=1.0,            # Euler–Maruyama 步长 / 钉距
    mode='sde',                   # 'sde'（扩散极限）| 'bean'（离散豆子）
    crowd_beta=1.0,               # 拥挤耦合强度 β
    es=dict(lr=0.02, sigma=0.05, pop=64, wd=1e-4,
            gens=600, batch=256, sigma_decay=0.9995, seed=42),
    data_seed=0, split_seed=42,
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(CFG['split_seed']); np.random.seed(CFG['split_seed'])

# ----------------------------------------------------------
# 1  数据（合成高斯簇；换 Drive 数据见注释）
# ----------------------------------------------------------
def make_blobs(F, K, N, seed):
    g = torch.Generator().manual_seed(seed)
    mus = 2.5 * torch.randn(K, F, generator=g)      # 每类一个均值簇
    y = torch.arange(N) % K
    y = y[torch.randperm(N, generator=g)]
    X = torch.randn(N, F, generator=g) + mus[y]
    return X, y

# 换成自己的数据（Colab）：
# from google.colab import drive; drive.mount('/content/drive')
# X = torch.load('/content/drive/MyDrive/Datasets/F_data_X.pt').float()
# y = torch.load('/content/drive/MyDrive/Datasets/F_data_Y.pt').long()

X, y = make_blobs(CFG['F'], CFG['K'], CFG['N'], CFG['data_seed'])

def stratified_split(y, seed):
    g = torch.Generator().manual_seed(seed)
    tr, va, te = [], [], []
    for k in y.unique():
        idx = (y == k).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(len(idx), generator=g)]
        n = len(idx); a, b = int(0.6 * n), int(0.8 * n)
        tr += [idx[:a]]; va += [idx[a:b]]; te += [idx[b:]]
    return torch.cat(tr), torch.cat(va), torch.cat(te)

i_tr, i_val, i_te = stratified_split(y, CFG['split_seed'])
mu, sd = X[i_tr].mean(0), X[i_tr].std(0).clamp_min(1e-6)
X = (X - mu) / sd
Xtr, ytr = X[i_tr].to(device), y[i_tr].to(device)
Xval, yval = X[i_val].to(device), y[i_val].to(device)
Xte, yte = X[i_te].to(device), y[i_te].to(device)
print(f'train {len(Xtr)} / val {len(Xval)} / test {len(Xte)}，'
      f'{CFG["K"]} 类，随机准确率 = {1 / CFG["K"]:.3f}')

# ----------------------------------------------------------
# 2  模型：钉板 = 输入条件的随机场，粒子 = 持久的 bean
# ----------------------------------------------------------
class GaltonSwarm(nn.Module):
    """前向即物理过程：P 个 bean 从板顶同口落入，逐排遭遇钉子被偏转；
    群密度 ρ 回授钉子（拥挤侧向力）；终点按类别格计数归一化。
    全程 torch.no_grad() —— 没有任何东西属于计算图。"""
    def __init__(self, F, K, P, T, G, M, dt, Delta, mode, crowd_beta):
        super().__init__()
        self.F, self.K, self.P, self.T, self.G, self.M = F, K, P, T, G, M
        self.dt, self.Delta, self.mode, self.beta = dt, Delta, mode, crowd_beta
        self.s_min = -(G - 1) / 2 * Delta
        self.s_max = (G - 1) / 2 * Delta
        self.bin_w = (self.s_max - self.s_min) / K
        self.register_buffer('centers', torch.linspace(
            self.s_min + self.bin_w / 2, self.s_max - self.bin_w / 2, K))
        # 全部可训练量打成一个一维向量供 NES 进化；声明为 buffer，
        # 因为它本来就不该出现在任何计算图里
        n_theta = F * M + G * T + G * T * M
        theta = torch.randn(n_theta) * 0.01
        theta[F*M : F*M + G*T] = 0.0     # 钉子初始无偏：p=0.5，最大发散（最大熵板）
        self.register_buffer('theta', theta.to(device))

    def unpack(self):
        F, M, G, T = self.F, self.M, self.G, self.T
        W = self.theta[:F*M].view(F, M)             # 编码器：x → 上下文 φ
        A = self.theta[F*M : F*M + G*T].view(T, G)  # 钉子偏置（排 t, 列 g）
        C = self.theta[F*M + G*T:].view(T, G, M)    # 钉子上下文权重
        return W, A, C

    def _peg_prob(self, A, C, phi, t):
        # 钉子在上下文 φ 下向右偏的概率：sigmoid(a + c·φ)
        return torch.sigmoid(A[t].unsqueeze(0) + phi @ C[t].T)      # (B, G)

    def forward(self, x, probe=False):
        with torch.no_grad():
            B = x.size(0)
            P, T, G, dt, d = self.P, self.T, self.G, self.dt, self.Delta
            W, A, C = self.unpack()
            phi = x @ W                                               # (B, M)
            s = torch.zeros(B, P, device=x.device)                    # 全部 bean 从顶端同口落入
            traj = [s.clone()] if probe else None
            rhos = [] if probe else None
            for t in range(T):
                c = (s / d + (G - 1) / 2).clamp(0.0, G - 1)           # 连续坐标 → 钉列
                c0 = c.floor().long().clamp(0, G - 2)
                w1 = c - c0.float()                                   # 线性插值权重
                p = self._peg_prob(A, C, phi, t)                      # (B, G)
                pl = (1 - w1) * p.gather(1, c0) + w1 * p.gather(1, c0 + 1)  # (B, P)

                # 群密度 ρ：三角核 soft 直方图，归一化为质量分布
                rho = torch.zeros(B, G, device=x.device)
                rho.scatter_add_(1, c0, 1 - w1)
                rho.scatter_add_(1, c0 + 1, w1)
                rho = rho / P
                # 拥挤侧向力：右侧密度高于左侧 → bean 被往左推（顺密度梯度下坡）
                grad = rho.gather(1, c0 + 1) - rho.gather(1, c0)      # (B, P)
                crowd = -self.beta * d * grad

                if self.mode == 'bean':                               # 离散豆子：钉子的伯努利结果
                    step = (torch.rand_like(s) < pl).float()
                    s = s + ((2 * step - 1) * d + crowd) * dt
                else:                                                 # sde：钉板二项发散的扩散极限
                    mu = d * (2 * pl - 1) + crowd
                    sig = 2 * d * torch.sqrt(pl * (1 - pl) + 1e-8)
                    s = s + mu * dt + sig * math.sqrt(dt) * torch.randn_like(s)
                s = s.clamp(self.s_min, self.s_max)

                if probe:
                    traj.append(s.clone())
                    rhos.append(rho.clone())

            # 读出 = 计数：终点位置按类别格 soft 直方图，粒子计数归一化
            sig_b = 0.75 * self.bin_w
            d2 = (s.unsqueeze(-1) - self.centers.view(1, 1, -1)) ** 2 / (2 * sig_b ** 2)
            hist = torch.softmax(-d2, dim=2).mean(dim=1)              # (B, K)
            logp = torch.log(hist + 1e-8)

            if not probe:
                return logp
            evo = None
            for s_t in traj:                                          # 逐时证据流 (B, T+1, K)
                d2t = (s_t.unsqueeze(-1) - self.centers.view(1, 1, -1)) ** 2 / (2 * sig_b ** 2)
                ht = torch.softmax(-d2t, dim=2).mean(dim=1, keepdim=True)
                evo = ht if evo is None else torch.cat([evo, ht], dim=1)
            return logp, dict(traj=torch.stack(traj, dim=2),        # (B, P, T+1)
                              rho=torch.stack(rhos, dim=2),          # (B, G, T)
                              hist=hist, evo=evo)                    # (B,K) / (B,T+1,K)

def fitness(model, Xb, yb, wd):
    """NES 的适应度 = -(NLL + 权重衰减)。只前向，永不反传。"""
    logp = model(Xb)
    nll = -logp.gather(1, yb.view(-1, 1)).mean()
    return -(nll + wd * model.theta.pow(2).sum())

# ----------------------------------------------------------
# 3  训练：(μ,λ)-NES 进化策略 —— 全局零梯度下降
# ----------------------------------------------------------
def centered_ranks(x):
    r = torch.empty_like(x)
    r[x.argsort()] = torch.arange(len(x), dtype=x.dtype, device=x.device)
    return (r - (len(x) - 1) / 2) / len(x)

class NES:
    """OpenAI-ES 风格：反对称采样 + 居中秩效用，只操作 theta。"""
    def __init__(self, theta, lr, sigma, pop, wd, seed, sigma_decay=1.0):
        # 持有自己的克隆：评估子代时 model.theta 会被逐个子代覆盖，
        # 若与 model 共享同一块内存，tell 的更新就会加在最后一个子代上
        self.theta, self.lr, self.wd = theta.clone(), lr, wd
        self.sigma, self.decay, self.pop = sigma, sigma_decay, pop
        self.half = pop // 2
        self.gen = torch.Generator(device='cpu').manual_seed(seed)

    def ask(self):
        eps = torch.randn(self.half, self.theta.numel(), generator=self.gen).to(self.theta.device)
        self.eps = eps
        delta = self.sigma * eps
        return torch.cat([self.theta + delta, self.theta - delta])

    def tell(self, fits):
        u = centered_ranks(fits)
        g = (u[:self.half] - u[self.half:]) @ self.eps   # 反对称采样对消基线
        self.theta.add_(self.lr / (2 * self.half * self.sigma) * g)
        self.sigma *= self.decay

@torch.no_grad()
def evaluate(model, X, y, repeats=4):
    """多次随机前向取均值：输出本身就是统计量。采样在推理时保持开启。"""
    probs = torch.stack([model(X).exp() for _ in range(repeats)]).mean(0)
    logp = probs.log()
    acc = (logp.argmax(1) == y).float().mean().item()
    nll = -logp.gather(1, y.view(-1, 1)).mean().item()
    single = [(model(X).argmax(1) == y).float().mean().item() for _ in range(repeats)]
    return acc, nll, float(np.std(single))

model = GaltonSwarm(CFG['F'], CFG['K'], CFG['P'], CFG['T'], CFG['G'], CFG['M'],
                    CFG['dt'], CFG['Delta'], CFG['mode'], CFG['crowd_beta']).to(device)

es = CFG['es']
nes = NES(model.theta, es['lr'], es['sigma'], es['pop'], es['wd'],
          es['seed'], es['sigma_decay'])
g = torch.Generator(device='cpu').manual_seed(CFG['split_seed'] + 1)
best_val, best_theta = -1.0, model.theta.clone()
history = dict(gen=[], tr_nll=[], val_nll=[], val_acc=[])
t0 = time.time()
for gen_i in range(1, es['gens'] + 1):
    pop = nes.ask()
    idx = torch.randint(0, len(Xtr), (es['batch'],), generator=g).to(device)  # 同批比较，减方差
    fits = []
    for i in range(nes.pop):
        model.theta.copy_(pop[i])
        fits.append(fitness(model, Xtr[idx], ytr[idx], nes.wd))
    nes.tell(torch.stack(fits))
    model.theta.copy_(nes.theta)
    if gen_i == 1 or gen_i % 20 == 0:
        va, vn, vs = evaluate(model, Xval, yval)
        history['gen'].append(gen_i); history['tr_nll'].append(-fits[0].item())
        history['val_nll'].append(vn); history['val_acc'].append(va)
        print(f'Gen{gen_i:04d}  σES {nes.sigma:.4f}  trNLL {history["tr_nll"][-1]:.4f}  '
              f'valNLL {vn:.4f}  valAcc {va:.4f}±{vs:.4f}  ({time.time()-t0:.0f}s)')
        if va > best_val:
            best_val, best_theta = va, model.theta.clone()

model.theta.copy_(best_theta)
print(f'>>> 训练完成，用时 {time.time()-t0:.0f}s，最佳 val acc {best_val:.4f}')

# ----------------------------------------------------------
# 4  训练曲线 + 测试（输出是分布：重复前向给出均值±标准差）
# ----------------------------------------------------------
plt.figure(figsize=(11, 3.5))
plt.subplot(1, 2, 1); plt.plot(history['gen'], history['val_nll']); plt.title('val NLL')
plt.subplot(1, 2, 2); plt.plot(history['gen'], history['val_acc']); plt.title('val acc')
plt.tight_layout(); plt.show()

te_acc, te_nll, te_std = evaluate(model, Xte, yte, repeats=8)
chance = 1 / CFG['K']
print(f'\n测试集：acc {te_acc:.4f}（单次前向离散度 ±{te_std:.4f}，随机 = {chance:.3f}）  NLL {te_nll:.4f}')

# ----------------------------------------------------------
# 5  探针：轨迹 / 群密度 / 逐时证据流 —— 模型内部现在可读
# ----------------------------------------------------------
logp_val = model(Xval)
pred_val = logp_val.argmax(1)
idxs = []
for c in (0, 1):   # 取两个被正确分类的样本做演示
    hit = ((yval == c) & (pred_val == c)).nonzero(as_tuple=True)[0]
    idxs.append(hit[0].item() if len(hit) else (yval == c).nonzero(as_tuple=True)[0][0].item())
xb = Xval[idxs]
_, pb = model(xb, probe=True)
traj = pb['traj'].cpu().numpy()          # (2, P, T+1)
hist = pb['hist'].cpu().numpy()          # (2, K)
evo = pb['evo'].cpu().numpy()            # (2, T+1, K)
rho = pb['rho'].cpu().numpy()            # (2, G, T)
T = CFG['T']

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for r, ax in enumerate(axes[0]):
    ax.plot(traj[r, :200].T, lw=0.6, alpha=0.08)          # 200 条 bean 轨迹
    ax.axhline(0, color='k', lw=0.3)
    ax.set_title(f'样本{r}（真类 {yval[idxs[r]].item()}）的 bean 轨迹')
    ax.set_xlabel('钉板排 t'); ax.set_ylabel('横向位置 s')
axes[1, 0].bar(np.arange(CFG['K']) - 0.2, hist[0], width=0.4, label='样本0')
axes[1, 0].bar(np.arange(CFG['K']) + 0.2, hist[1], width=0.4, label='样本1')
axes[1, 0].set_title('终点类别格计数（读出=直方图）'); axes[1, 0].legend()
top = hist[0].argsort()[::-1][:3]
for i, k in enumerate(top):
    axes[1, 1].plot(evo[0, :, k], label=f'样本0→类{k}', lw=2 - 0.4 * i)
top = hist[1].argsort()[::-1][:3]
for i, k in enumerate(top):
    axes[1, 1].plot(evo[1, :, k], '--', label=f'样本1→类{k}', lw=2 - 0.4 * i)
axes[1, 1].set_title('逐时证据流：哪个格子何时开始 accumulation')
axes[1, 1].set_xlabel('钉板排 t'); axes[1, 1].legend(fontsize=7)
plt.tight_layout(); plt.show()

plt.figure(figsize=(4, 3))
plt.imshow(rho[0], aspect='auto', origin='lower', cmap='magma')
plt.title('群密度 ρ（样本0）：粒子改造自己的介质'); plt.xlabel('t'); plt.ylabel('钉列')
plt.colorbar(); plt.tight_layout(); plt.show()

# ----------------------------------------------------------
# 6  保存
# ----------------------------------------------------------
torch.save(dict(model_class='GaltonSwarm', model_args=(CFG['F'], CFG['K'], CFG['P'], CFG['T'],
                    CFG['G'], CFG['M'], CFG['dt'], CFG['Delta'], CFG['mode'], CFG['crowd_beta']),
                theta=model.theta.cpu(), mu=mu, sd=sd, config=CFG, history=history),
           'galton_swarm.pt')
# torch.save(..., '/content/drive/MyDrive/.../galton_swarm.pt')  # Colab 上同时存 Drive
print('>>> 已保存 galton_swarm.pt')
print('>>> 全文 loss.backward() 调用次数：0（全局无梯度下降）')
