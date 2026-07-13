# PyMIC Example Reproduction

这个仓库记录我对 [PyMIC](https://github.com/HiLab-git/PyMIC) 官方示例的复现过程，重点保存可重复执行的配置、终端命令、实验日志、评价结果和对比图。

> 本仓库是个人学习与复现记录，不是 PyMIC 官方仓库。原始代码和示例分别来自 [HiLab-git/PyMIC](https://github.com/HiLab-git/PyMIC) 与 [HiLab-git/PyMIC_examples](https://github.com/HiLab-git/PyMIC_examples)。

## 已完成实验

| Example | Task | Device | Status | Best accuracy | Best AUC |
|---|---|---|---|---:|---:|
| [AntBee](experiments/antbee/README.md) | ResNet18 二分类迁移学习 | Apple MPS | 完成 | 94.77% | 97.90% |

AntBee 分别比较了两种迁移学习策略：

- CE1：微调 ResNet18 全部参数。
- CE2：冻结骨干网络，只微调最后的分类层。

![AntBee comparison](experiments/antbee/figures/fig_ce1_ce2_comparison.png)

## 仓库结构

```text
Pymic_example/
├── experiments/
│   └── antbee/
│       ├── config/       # 训练与评价配置
│       ├── figures/      # 实验图表及绘图脚本
│       ├── logs/         # 完整训练日志
│       ├── results/      # 预测概率和指标汇总
│       ├── scripts/      # 数据清单与 ROC 脚本
│       └── README.md     # 逐步复现记录
├── patches/              # Apple Silicon / MPS 兼容补丁
├── requirements.txt      # 本次验证的主要环境版本
└── README.md
```

## 复现环境

- macOS，Apple Silicon
- Python 3.13.9
- PyTorch 2.11.0
- torchvision 0.26.0
- MPS GPU 后端

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

详细命令和实验说明见 [AntBee 复现记录](experiments/antbee/README.md)。

## 数据与模型文件

数据集和训练 checkpoint 不提交到 Git：

- Hymenoptera 数据集请从 PyTorch 官方地址下载。
- ResNet18 预训练权重由 torchvision 自动下载。
- 本地训练 checkpoint 通过 `.gitignore` 排除，避免仓库体积过大。

## 当前限制

- 当前结果使用官方示例中的验证集作为测试输入，并非独立测试集。
- 每种设置只运行了一次，尚未进行多随机种子统计。
- 当前只完整验证了 PyMIC 分类流程在 Apple MPS 上的运行；分割及其他训练模式需要继续适配和验证。
