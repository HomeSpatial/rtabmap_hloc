# rtabmap_hloc

独立 Python 包，将 RTAB-Map 数据库导出为图像、相机与 COLMAP 稀疏模型。安装包名为 `rtabmap-hloc-reprocess`，Python 导入名为 `rtabmap_hloc_reprocess`，命令为 `rtabmap-hloc`。

当前是开源候选，项目许可证尚待权属人确认，尚未发布。默认依赖官方 HLOC 和公开 pycolmap，数据库读取不需要 RTAB-Map C++。

## 安装与使用

Linux、Python 3.12，使用本仓库自己的锁文件：

```bash
uv sync --locked
uv run --locked rtabmap-hloc --help
uv run --locked rtabmap-hloc ./input.db -o ./output
```

默认使用 ALIKED、MegaLoc、LightGlue 做完整 SfM，再通过 Sim3 对齐恢复尺度。`--use-existing-poses` 是另一条使用现有位姿进行三角化的路径，不应将它的结果称为纯视觉 SfM。具体内参、对齐与特征选项参见命令帮助。

本仓库不依赖 remap；导出的图像、标定和 COLMAP 模型可供独立安装的 remap 或其他重建工具使用。与 remap 的 RWp 输入清单格式不同，不能直接把输出目录当作清单。

官方 [HLOC](https://github.com/cvg/Hierarchical-Localization) revision 固定在 `pyproject.toml`，传递依赖由 `uv.lock` 固定；[pycolmap](https://github.com/colmap/colmap) 使用公开 Python 包。模型权重不随本仓库分发，首次使用特征模型可能下载权重；代码及权重的使用条件分别以上游为准。

## 宿主项目使用自己的 HLOC

将本包保留为宿主 uv workspace 的成员，在宿主根配置依赖来源即可覆盖本包默认来源，不在 Python 代码中探测宿主目录：

```toml
[tool.uv.sources]
hloc = { path = "3rdparty/hloc", editable = true }
```

宿主的 `uv.lock` 固定实际 fork 与其他依赖组合，本仓库的锁文件不会自动控制宿主环境。普通 pip 不读取 uv 来源覆盖；使用 fork 的环境应由宿主统一安装。原生 pycolmap 也由宿主负责安装、版本声明与导入路径。

## 开发

```bash
uv sync --locked --extra dev
uv run --locked --extra dev python -m pytest tests
```

`src/rtabmap_hloc_reprocess` 保存运行代码，`tests` 保存测试。通用修改在本仓库提交；宿主验证后更新对应子模块指针。不要把私有数据、实验结果或内部服务配置提交到公共仓库。
