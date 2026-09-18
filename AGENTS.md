# AGENTS.md — Bristol

## 项目概述

**课题：高尔顿板群智网络（Galton Swarm Network）** —— 用受随机过程支配的粒子群体（"bean"）取代梯度回归的神经网络研究。

核心愿景：信息落入网络 → 群体在"集体智能"决策下分流（钉板偏转 + 拥挤耦合）→ 终点计数给出统计结果。**训练全程无梯度下降**（NES 进化策略），随机过程即计算本身。

## 当前状态（2026-09-19）

- 首轮实验完成（合成数据）：**val acc 0.9619**（14 类，随机 0.071），详见 `reports/2026-09-19_galton_swarm_v1.md`，tag `exp-swarm-v1`
- 关键发现：分类决策主要发生在末段钉排（t≈15–24）；读出=计数成立，内部状态可读且携带分类信息
- 待办：测试集 acc 待补记；确认 Colab 硬件（3569s 偏慢，疑似 CPU runtime）；下轮换真实 Drive 数据
- 旧 MDU 审计结论见 `legacy/galton_sde_fixed.py`（仅存档，勿运行）

## 目录结构

```
Bristol/
├── galton_swarm.py        # 当前原型：GaltonSwarm + NES 训练 + 探针可视化（自上而下脚本，Colab 直接 %run）
├── reports/               # 每轮实验记录：配置 X / 结果 Y / 结论 Z
├── legacy/
│   └── galton_sde_fixed.py # 旧 MDU 遗留代码，仅存档参考，勿直接运行（数据路径指向 Drive 且含审计出的结构缺陷）
└── requirements.txt       # torch>=2.1, numpy>=1.24, matplotlib>=3.7（Colab 预装）
```

## 运行方式

**本地 Windows 机器没有 Python/PyTorch，不能运行训练；只能做静态检查。一切执行在 Colab。**

Colab 首格：

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/Jakoco/Bristol.git
%cd Bristol
!pip install -r requirements.txt
# 重连时：%cd Bristol && !git pull
```

训练：`%run galton_swarm.py`（GPU 约 1–2 分钟，600 代 NES）。冒烟：把 `CFG['es']['gens']` 调到 5–10。

换真实数据：改 `galton_swarm.py` 第 1 节注释处（Drive 的 `F_data_X/F_data_Y`），替换 `make_blobs` 调用即可，分割/标准化代码复用。

## 设计准则（对应旧 MDU 审计结论）

1. **全局无梯度下降**：训练只用 NES（反对称采样 + 居中秩），全文零 `loss.backward()`；参数声明为 buffer，物理上进不了计算图。
2. **读出 = 计数**：类别分布是终点位置的 soft 直方图，禁止线性投影头（无 slots/einsum）。
3. **粒子是持久实体，采样全程开启**：训练/推理都不去噪（旧 MDU 的 `noise.zero_()` 是死刑点）。
4. **交互 = 介质耦合**：群密度 ρ 逐排回授钉子偏置（拥挤侧向力）；禁止用"特征图"冒充智能体。
5. **可读性内建**：`probe()` 输出轨迹、密度场、逐时证据流；每轮实验必须保存并查看。

## 关键超参（CFG）

`P=256` 粒子数 / `T=24` 钉排数 / `G=33` 每排钉列 / `M=16` 上下文维度 / `mode='sde'|'bean'`。
参数总量 ≈ 13.6k（编码器 + 钉子偏置 + 钉子上下文权重）。

## 维护原则

- 每轮 Colab 实验后：结果结论记入本文件"当前状态"与 `reports/`（建目录后），例：配置 X → 结果 Y → 结论 Z
- 改动训练/读出/交互结构 = 大改动，先更新本文件的设计准则再写代码
- `legacy/` 下的旧代码不要修复、不要运行，只作历史对照
- 只放稳定具体的知识：命令给全、参数给值，不给"运行训练脚本"这种空话
