# RTABMap-HLOC

[English](README.md) · [高级安装与集成](docs/advanced.md) · [输出格式](docs/outputs.md)

将 RTAB-Map 数据库或 `rtabmap-export` 目录中的图像、标定和位姿，转换为
**COLMAP 稀疏重建模型**，并通过 Sim3 对齐恢复输入轨迹的米制尺度。
成功后生成可移交的 `reconstruction.json`，下游不需要连接 RTAB-Map 服务。
本工具不生成稠密网格或结构化建筑模型。

当前版本为 **0.1.0 / Alpha**，支持基线为 Linux、Python 3.12。
读取现有数据库无需安装 RTAB-Map C++。

## 快速开始

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 和 Git 后：

```bash
git clone https://github.com/HomeSpatial/rtabmap_hloc.git
cd rtabmap_hloc
uv sync --locked
uv run --locked rtabmap-hloc --version
uv run --locked rtabmap-hloc ./input.db -o ./output --scene-id house
```

安装包名为 `rtabmap-hloc-reprocess`，Python 导入名为 `rtabmap_hloc_reprocess`，
命令为 `rtabmap-hloc`，也支持 `python -m rtabmap_hloc_reprocess`。
依赖包含 PyTorch，安装体积较大。神经网络特征推理建议使用 CUDA GPU，
具体可用性取决于 PyTorch 和驱动。首次运行可能下载模型权重；单元测试无需 GPU 或权重。

默认使用 **ALIKED + MegaLoc + LightGlue** 进行完整 SfM，再通过 Sim3 对齐到
RTAB-Map 轨迹。默认固定 RTAB-Map 的逐帧相机内参。
对齐成功只说明恢复了输入轨迹的尺度，不代表通过独立的几何精度验收。

## 常用方式

```bash
# 输入导出目录，包含 rgb/、calib/、poses.txt。
uv run --locked rtabmap-hloc ./exported -o ./output-exported

# 使用输入位姿进行三角化，区别于完整视觉 SfM。
uv run --locked rtabmap-hloc ./input.db -o ./output-poses --use-existing-poses

# 允许优化导入的相机内参。
uv run --locked rtabmap-hloc ./input.db -o ./output-refined --refine-intrinsics

# 写入场景 bundle 的独立阶段。
uv run --locked rtabmap-hloc ./input.db --bundle ./house --stage sfm

uv run --locked rtabmap-hloc --help
```

源数据库默认以只读方式打开，导入的照片复制到输出目录。
已包含 `reconstruction.json` 的目录不允许原地重跑，请换新目录。
`--up-axis` 默认为 `+Z`，只声明世界坐标约定，不旋转几何。

## 输出与失败状态

| 路径 | 内容 |
| --- | --- |
| `images/` | 导出的输入照片 |
| `hloc/` | 特征、检索、匹配和中间 SfM |
| `sfm_aligned/` | 已对齐的 COLMAP 二进制稀疏模型 |
| `metadata.json` | 配置、内参策略、重建指标 |
| `alignment-report.json` | 对齐方法与残差 |
| `reconstruction.json` | 已注册照片和原生资产的路径、坐标与哈希 |
| `run-status.json` | 执行、对齐、质量状态 |

重建、对齐、导出或 bundle 发布失败时返回非零。
对齐失败不发布新的描述；成功也保留质量状态 `not_evaluated`。
照片集合只声明已注册照片，中间输出可以保留未注册照片。
具体字段、断点续跑约束和下游消费方法参见[输出说明](docs/outputs.md)。

## 开发与验证

```bash
uv sync --locked --extra dev
uv run --locked --extra dev python -m pytest
uv run --locked --extra dev python scripts/check_release.py
uv run --locked --extra dev python -m build
uv run --locked --extra dev python -m twine check dist/*
uv run --locked --extra dev python scripts/check_release.py --dist dist
```

贡献、发行流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。CI 使用公开依赖，
验证源码、wheel、sdist 和离开源码目录后的安装入口。

## 集成边界

本包独立运行不需要 remap。`--inference-backend colab` 是可选宿主接口，
需要另行提供兼容的 `remap.colab.model_backend`，安装本包不会自动得到该服务。
`onnx` extra 仅安装运行时，不会给官方 HLOC 增加 ONNX 模型配置。
宿主覆盖 HLOC 来源的方法见[高级说明](docs/advanced.md)。

锁文件固定 Python 依赖与 HLOC revision，但上游 Torch Hub 仍可能在运行时获取代码和权重，
因此不能把它视为完全离线、完整冻结的模型环境。
真实场景的精度、显存占用和性能需要另外验证。
仓库不包含私人扫描、照片、模型权重或重建结果。

## 许可

自有代码与文档采用 [Apache-2.0](LICENSE)。共享交换格式参考实现的归属见
[NOTICE](NOTICE)，外部依赖和模型权重继续遵循各自许可，详见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
