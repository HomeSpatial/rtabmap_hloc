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

本仓库不依赖 remap；导出的图像、标定和 COLMAP 模型可供独立安装的 remap 或其他重建工具使用。成功后自动在输出目录生成 `reconstruction.json`（reconstruction description v1），可直接交给 `remap reconstruct`；无须重新填写照片和相机路径。

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

## 可移交的结果与失败状态

```bash
rtabmap-hloc input.db -o scene/sfm --scene-id house
remap reconstruct scene/sfm/reconstruction.json --out scene/dense
```

描述引用本目录中 `sfm_aligned` 的 COLMAP 二进制、已注册照片、metadata 和对齐报告，并包含文件哈希、米制坐标和照片覆盖关系。不要求下游安装 HLOC 或连接宿主服务。`--up-axis` 默认为 RTAB-Map 常用的 `+Z`；它仅声明坐标约定，不旋转几何。非标准采集坐标须显式指定。

`run-status.json` 分别记录执行、对齐和质量状态。RANSAC 和最小二乘对齐均失败、变换非法、无注册照片或描述导出失败时，命令返回非零，不发布新的 `reconstruction.json`。`--use-existing-poses` 标记为 `input_poses`；估计成功也只代表完成对齐，质量仍为 `not_evaluated`，不能当作尺度精度验收。

已发布 `reconstruction.json` 的目录不可原地重跑，以免破坏已交接文件和哈希；重跑请选择新目录。尚未发布描述的旧中间结果仍可通过 `--skip-image-export --skip-hloc` 继续处理。

一个 scene bundle 可以引用多个阶段的描述。每次运行保留独立目录和明确的资产 ID；多个稀疏模型并存时，下游必须明确选择对应的相机和照片，不能默认采用最后一个结果。添加 `--bundle house` 可自动发布到同一 bundle；默认阶段 sfm、工作目录 house/work/sfm、快照 house/stages/sfm，scene_id 默认取 house。已有 bundle 自动继承 scene_id。使用 `--stage sfm-r2` 保留另一份结果，禁止覆盖旧阶段。

```bash
rtabmap-hloc input.db --bundle house
remap reconstruct --bundle house
```

发布失败时命令返回非零，独立结果保留；可用 `recon bundle append house --result <work>/reconstruction.json --stage sfm --operation rtabmap-hloc` 重试发布。
