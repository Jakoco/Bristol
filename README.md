[工作流说明书.md](https://github.com/user-attachments/files/32390014/default.md)
# Kimi Code 云端研究工作流说明书

> 适用场景：本地电脑性能弱（跑不了 PyTorch），训练在 Colab，代码由 Kimi Code 管理。
> 角色分工：**本地电脑 = 指挥官，Kimi Code = 工程师，GitHub = 仓库，Colab = 算力，Google Drive = 存档。**

---

## 一、架构总览

```
本地 Windows（PowerShell + Kimi Code CLI）
        │  git push / pull
        ▼
GitHub 仓库（代码唯一中转站 + 版本历史）
        ▼
Colab（GPU 训练，每次开机 git clone/pull）
        │  checkpoint / 日志
        ▼
Google Drive（断点续训、实验结果，唯一持久化存储）
```

**铁律**：
1. 代码只信 GitHub——本地和 Colab 都可能丢，推上去才算数。
2. Checkpoint 只信 Drive——Colab 的本地磁盘随时被回收。
3. 每轮实验结论写进 AGENTS.md——它是 Kimi Code 的"项目记忆"。

---

## 二、一次性配置（已完成 ✔）

| 项目 | 状态 |
|---|---|
| Kimi Code CLI 安装（Windows） | ✔ |
| KIMI_SHELL_PATH 指向 D 盘 Git Bash | ✔ |
| Git for Windows | ✔ |
| Git 用户身份（user.name / user.email） | ✔ |
| GitHub 账号 + 空仓库 Bristol | ✔ |
| 首次 push 成功（AGENTS.md 已上库） | ✔ |
| Colab 接收 notebook 模板 | ✔ |

Colab 首格模板：
```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/Jakoco/Bristol.git
%cd Bristol
!pip install -r requirements.txt   # 有 requirements.txt 后启用
```

---

## 三、日常开发循环（核心流程）

### 1. 本地写代码
```powershell
cd F:\Kimi_project\Bristol
kimi
```
对 Kimi Code 用自然语言下任务，例如：
> "把 data_loader.py 改成支持按年份划分训练/测试集，并在 AGENTS.md 里记录这个决定"

它会自己改文件、更新 AGENTS.md。**涉及训练验证的改动，让它保持小步提交。**

### 2. 提交推送（在 Kimi Code 界面里输）
```
! git add -A && git commit -m "描述这次改动" && git push
```
或者直接对它说："提交并推送本次改动"。

### 3. Colab 执行
- 新开/重连 Colab → 跑首格模板（重连只需 `%cd Bristol && !git pull`）
- 启动训练（长训练用 `nohup python train.py > logs/train.log 2>&1 &` 挂后台）
- checkpoint 路径一律指向 `/content/drive/MyDrive/...`

### 4. 报错回流
Colab 报错 → 复制完整 traceback → 贴给 Kimi Code："Colab 上跑 train.py 报这个错，修一下并推送"
→ Colab `!git pull` → 重跑。

### 5. 实验记录
每轮实验结束，对 Kimi Code 说：
> "本轮实验：配置 X，结果 Y，结论 Z，记入 AGENTS.md 和 reports/"

---

## 四、新建项目流程

复用同一套架构，只是换个仓库：

```powershell
# 1. GitHub 网页建空仓库（不勾 README），记下名字，如 solar-pinn
# 2. 本地：
mkdir F:\Kimi_project\solar-pinn
cd F:\Kimi_project\solar-pinn
kimi
# 3. 界面里：
/init
# 4. 首次提交（注意换仓库名；safe.directory 同样要加一次）：
! git config --global --add safe.directory F:/Kimi_project/solar-pinn
! git init && git add -A && git commit -m "init" && git branch -M main && git remote add origin https://github.com/Jakoco/solar-pinn.git && git push -u origin main
# 5. Colab 首格模板换仓库地址即可
```

建议每个研究课题一个仓库，不要全塞在 Bristol 里。

---

## 五、维护指南

### 每周维护
- **看额度**：Kimi Code 界面输 `/usage`；控制台看周额度和 5 小时滚动窗口（80% 时有加油包入口）
- **更新 CLI**：界面输 `/exit` 回到 PowerShell，跑 `kimi upgrade`

### 每月维护
- 清理 Drive 里过期的 checkpoint（只留最优 + 最近一轮）
- 检查 AGENTS.md 是否还反映项目现状（大了就让它精简）
- `git log --oneline` 回顾提交历史，确认实验可复现（每个重要实验对应一个 commit/tag）

### AGENTS.md 维护原则
- 只放**稳定的**项目知识：目录结构、运行命令、编码约定、关键实验结论
- 放**具体的**：`python train.py --config configs/base.yaml` 而不是"运行训练脚本"
- 每轮重要实验后让它更新；发现它引用过时的内容立即纠正

### 会话策略（对抗 Colab 易失）
- 单会话内任务拆小，做完即 push
- 长训练绝不交给 Kimi Code 在交互会话里跑——你自己 `nohup` 挂后台
- config 一律默认 `resume=True`，断了从 Drive 的 checkpoint 续，不重来

---

## 六、故障排查速查表

| 症状 | 原因 | 解法 |
|---|---|---|
| `kimi` 命令找不到 | PATH 未生效 | 重开终端；查 kimi.exe 所在目录手动加 PATH |
| `无法将"kimi"项识别为...` | 同上等 | 同上 |
| dubious ownership | F 盘 + Git Bash 所有者校验 | `git config --global --add safe.directory F:/Kimi_project/<项目名>` |
| Please tell me who you are | Git 身份未配 | `git config --global user.email/name` |
| Host key verification failed | SSH 主机密钥缺失 | 换 HTTPS remote（本说明书全部用 HTTPS） |
| push 要求登录 | 凭据过期 | 正常弹窗，浏览器授权一次即可 |
| Colab 断连后一切没了 | VM 被回收 | 正常现象：重跑首格模板，`git pull`，从 Drive checkpoint 续训 |
| 额度突然不能用 | 周额度/5小时窗口/月额度触顶 | `/usage` 查看；等刷新或开加油包 |
| Kimi Code 改了代码没效果 | Colab 没 pull | `%cd Bristol && !git pull` |

---

## 七、常用命令速查

**Kimi Code 界面内**：
| 命令 | 作用 |
|---|---|
| `/login` `/logout` | 登录/退出 |
| `/usage` | 查额度 |
| `/init` | 生成/更新 AGENTS.md |
| `/new` `/sessions` | 新会话 / 恢复历史会话 |
| `/compact` | 上下文太长了压缩一下 |
| `/model` | 切换模型 |
| `! <命令>` | 直接跑 shell 命令 |
| 两次 Ctrl-C 或 `/exit` | 退出 |

**PowerShell 里**：
| 命令 | 作用 |
|---|---|
| `kimi` | 启动 |
| `kimi -p "任务"` | 单条指令不进入交互界面 |
| `kimi -C` | 继续上次会话 |
| `kimi upgrade` | 升级 CLI |

---

*最后更新：2026-09-18*
