# RDK S100：YOLOv5nu 从训练到部署

本手册旨在帮助具备基础 Linux 和命令行经验的开发者，快速完成基于 RDK S100 的模型训练、量化与板端部署。

![项目架构图](images/S0-01-project-architecture.jpg)

本篇完成从训练到 S100 部署的完整链路：训练 YOLOv5nu，建立 FP32 基线，导出六路 ONNX，用 OpenExplorer 3.7.0 生成 `yolov5nu_s100_nv12.hbm`，再在板端检测、评估精度和测试性能。开始前，请确认开发主机已安装 Linux 或 WSL、Conda、Docker、Git 和 `wget`。

## Stage 0：项目与基础环境

### 0.1 获取项目

执行位置：开发主机。

```bash
# 统一项目、工具链和下载目录，后文直接复用这些变量。
export S100_WORKSPACE="$HOME/s100-yolov5u-workspace"
export PROJECT_ROOT="$S100_WORKSPACE/s100_yolov5u_train2deploy"
export TOOLCHAIN_ROOT="$S100_WORKSPACE/toolchain"
export DOWNLOAD_ROOT="$TOOLCHAIN_ROOT/downloads"
export OE_ROOT="$TOOLCHAIN_ROOT/drobotics_s100_s600_open_explorer_v3.7.0"

# 创建工作区和大文件下载目录。
mkdir -p "$S100_WORKSPACE" "$DOWNLOAD_ROOT"
# 将代码、固定数据集、脚本和文档克隆到项目目录。
git clone git@github.com:Lutra-wang/s100_yolov5u_train2deploy.git "$PROJECT_ROOT"
```

仓库中与本阶段相关的目录如下：

```text
s100_yolov5u_train2deploy/
├── data/coco2017_val128_s100/  # 固定的训练、验证和测试数据
├── train/                     # Ultralytics 源码和训练输出
├── exchange/onnx/             # ONNX 交换模型
├── conversion/                # 校准图和 OE 编译输出
├── validation/                # 精度评估 JSON
├── scripts/                   # 评估与部署辅助脚本
└── docs/                      # 手册和证据文档
```

仓库不包含 OE 镜像和安装包；这些大文件在下一节下载。

### 0.2 配置 OpenExplorer

执行位置：开发主机。

```bash
# 支持断点续传，下载 OE 3.7.0 CPU 镜像包。
wget -c -P "$DOWNLOAD_ROOT" \
  https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe/3.7.0/ai_toolchain_ubuntu_22_s100_s600_cpu_v3.7.0.tar
# 下载与镜像版本一致的 OE 工具链包。
wget -c -P "$DOWNLOAD_ROOT" \
  https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe/3.7.0/oe-package-3.7.0-s100-s600.tgz

# 将 OE 镜像导入本机 Docker。
docker load -i "$DOWNLOAD_ROOT/ai_toolchain_ubuntu_22_s100_s600_cpu_v3.7.0.tar"
# 把工具链解压到独立目录，不与 Git 仓库混放。
tar -xf "$DOWNLOAD_ROOT/oe-package-3.7.0-s100-s600.tgz" -C "$TOOLCHAIN_ROOT"
```

本阶段只完成下载、镜像加载和工具链解压，暂不进入容器。预期 `$OE_ROOT/run_docker.sh` 已存在；若不存在，先确认压缩包是否完整以及解压目录是否正确。

## Stage 1：模型训练与导出

### 1.1 配置训练环境

执行位置：开发主机。

```bash
# 定义数据集、训练、Model Zoo 和 ONNX 路径。
export DATASET_ROOT="$PROJECT_ROOT/data/coco2017_val128_s100"
export TRAIN_ROOT="$PROJECT_ROOT/train"
export ULTRALYTICS_ROOT="$TRAIN_ROOT/ultralytics"
export MODEL_ZOO_ROOT="$PROJECT_ROOT/model_zoo_s"
export ONNX_ROOT="$PROJECT_ROOT/exchange/onnx"

# 获取经本案例验证的 Ultralytics 和 S100 Model Zoo 分支。
git clone --branch v8.4.152 --single-branch \
  https://github.com/ultralytics/ultralytics.git "$ULTRALYTICS_ROOT"
git clone --branch s100 --single-branch \
  https://github.com/D-Robotics/rdk_model_zoo_s.git "$MODEL_ZOO_ROOT"

# 创建 Python 3.10 训练环境，并在本手册中只进入一次。
conda create -y -n oe-s100-yolov5u python=3.10 pip
conda activate oe-s100-yolov5u
# 安装 CPU 版 PyTorch、项目版 Ultralytics 及评估/检查依赖。
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e "$ULTRALYTICS_ROOT" pycocotools onnx
```

后续 Stage 1 命令都在这个 Conda 环境中执行。

### 1.2 准备项目数据集

执行位置：开发主机 / Conda 环境。

本案例直接使用仓库内的固定数据集，不需要额外下载 COCO。`dataset.yaml` 定义训练数据路径，`images/test` 与 `annotations/instances_test.json` 用于固定测试集评估。

```bash
# 只列出本阶段必需的数据配置和标注，确认克隆内容完整。
ls -l "$DATASET_ROOT/dataset.yaml" \
  "$DATASET_ROOT/annotations/instances_test.json"
# 只把 YAML 顶层 path 改为当前克隆的数据集绝对路径。
sed -i "s|^path:.*|path: $DATASET_ROOT|" "$DATASET_ROOT/dataset.yaml"
```

仓库中的 YAML 保留了制作数据时的源路径，因此训练前必须将 `path:` 适配为当前 `$DATASET_ROOT`。如果任一文件不存在，请先更新项目仓库，不要自行改用其他数据集。

### 1.3 训练 YOLOv5nu

执行位置：开发主机 / Conda 环境。

`yolo` 是 Ultralytics 命令行工具，`detect train` 用于训练目标检测模型。

```bash
# 从官方 YOLOv5nu 预训练权重开始，使用项目数据配置训练。
yolo detect train \
  model=yolov5nu.pt data="$DATASET_ROOT/dataset.yaml" \
  imgsz=640 epochs=10 batch=8 workers=0 device=cpu \
  project="$TRAIN_ROOT/runs" name=coco2017_val128_s100 \
  seed=20260915 deterministic=True exist_ok=True

# exist_ok=True 将输出稳定在同一目录，后续命令不需要猜测数字后缀。
RUN="$TRAIN_ROOT/runs/coco2017_val128_s100"
```

本步生成 `$RUN/weights/best.pt`；这是后续评估和导出使用的最优权重。若该文件未生成，先回看训练终端的首个报错。

<!-- IMAGE_SLOT:S1-01 -->
> **待补图 S1-01｜训练完成结果**
> 建议文件：`docs/images/S1-01-training-complete.png`
> 截图内容：训练结束终端、best.pt 路径和关键指标。
> 图注：YOLOv5nu 训练完成并生成最优权重。

### 1.4 建立 FP32 精度基线

执行位置：开发主机 / Conda 环境。

```bash
# 固定测试集标注，确保 FP32 与后续 BPU 评估口径一致。
ANN="$DATASET_ROOT/annotations/instances_test.json"

# 用 best.pt 对测试图推理，输出 COCO 预测 JSON。
python "$PROJECT_ROOT/scripts/evaluate_coco_official.py" fp32-to-json \
  --weights "$RUN/weights/best.pt" \
  --images-dir "$DATASET_ROOT/images/test" \
  --annotations "$ANN" \
  --output "$PROJECT_ROOT/validation/quantization_fp32_predictions.json" \
  --imgsz 640 --conf 0.25 --iou 0.7
# 调用 pycocotools 计算并保存 FP32 指标。
python "$PROJECT_ROOT/scripts/evaluate_coco_official.py" evaluate \
  --annotations "$ANN" \
  --predictions "$PROJECT_ROOT/validation/quantization_fp32_predictions.json" \
  --metrics "$PROJECT_ROOT/validation/quantization_fp32_metrics.json"
```

指标写入 `validation/quantization_fp32_metrics.json`。仓库已有的 16 图实测记录为 mAP50 `0.388`、mAP50-90 `0.292`、mAP50-95 `0.266`；小样本结果用于本链路回归，不代表完整 COCO2017 基准。

<!-- IMAGE_SLOT:S1-02 -->
> **待补图 S1-02｜FP32 精度结果**
> 建议文件：`docs/images/S1-02-fp32-metrics.png`
> 截图内容：FP32 的三项 mAP 终端结果。
> 图注：固定测试集上的 FP32 精度基线。

### 1.5 导出 ONNX

执行位置：开发主机 / Conda 环境。

Ultralytics 默认导出面向通用推理；Model Zoo 的 `export_monkey_patch.py` 调整输出形式，使其匹配 S100 量化和部署代码。

```bash
# 导出 Model Zoo 适配的六输出 ONNX。
python "$MODEL_ZOO_ROOT/samples/Vision/ultralytics_yolo/x86/export_monkey_patch.py" \
  --pt "$RUN/weights/best.pt" --optse 11
# 使用固定交付名保存 ONNX。
mkdir -p "$ONNX_ROOT"
cp "$RUN/weights/best.onnx" "$ONNX_ROOT/yolov5nu_coco2017_val128_s100.onnx"
# 让 ONNX 检查器验证模型结构；成功时输出 onnx-ok。
python -c "import onnx; onnx.checker.check_model('$ONNX_ROOT/yolov5nu_coco2017_val128_s100.onnx'); print('onnx-ok')"
```

六路输出由三组特征层的 cls 和 bbox 组成：

| 特征层 | cls | bbox |
|---|---|---|
| s8 | `output0` `[1,80,80,80]` | `371` `[1,80,80,64]` |
| s16 | `379` `[1,40,40,80]` | `387` `[1,40,40,64]` |
| s32 | `395` `[1,20,20,80]` | `403` `[1,20,20,64]` |

其中 `80` 是 COCO 类别数，`64` 是 YOLOv5u 的 bbox 分布输出。

<!-- IMAGE_SLOT:S1-03 -->
> **待补图 S1-03｜ONNX 六路输出**
> 建议文件：`docs/images/S1-03-onnx-six-outputs.png`
> 截图内容：ONNX 的三组 cls/bbox 输出名称和形状。
> 图注：Model Zoo 适配导出的六路 ONNX 输出。

## Stage 2：模型量化与编译

### 2.1 配置量化环境

执行位置：先在开发主机准备目录和校准图，再进入 OE 容器。

```bash
# 重建 Stage 2 所需主机路径，便于从新终端继续执行。
export S100_WORKSPACE="$HOME/s100-yolov5u-workspace"
export PROJECT_ROOT="$S100_WORKSPACE/s100_yolov5u_train2deploy"
export TOOLCHAIN_ROOT="$S100_WORKSPACE/toolchain"
export OE_ROOT="$TOOLCHAIN_ROOT/drobotics_s100_s600_open_explorer_v3.7.0"
export DATASET_ROOT="$PROJECT_ROOT/data/coco2017_val128_s100"

# 创建校准与输出目录，并按文件名取训练集前 20 张图。
mkdir -p "$PROJECT_ROOT/conversion/cal_images" "$PROJECT_ROOT/conversion/output"
find "$DATASET_ROOT/images/train" -maxdepth 1 -name '*.jpg' | sort | head -20 \
  | xargs -r -I{} cp '{}' "$PROJECT_ROOT/conversion/cal_images/"

# 从 OE 工具链启动本手册唯一一次交互式量化容器。
cd "$OE_ROOT"
bash run_docker.sh "$PROJECT_ROOT" cpu
```

该命令使主机项目目录与容器 `/data` 共享同一组文件。从下一节开始，命令均在容器 `/data` 中执行。

### 2.2 准备校准集

执行位置：OE 容器。

2.1 已在主机从训练集按文件名排序选取 20 张图；由于目录共享，容器中可直接读取 `/data/conversion/cal_images`。校准集只用于确定 PTQ 数值范围，不参与模型训练。

```bash
# 输出应为 20；不足时先检查训练集路径和图片扩展名。
find /data/conversion/cal_images -maxdepth 1 -name '*.jpg' | wc -l
```

### 2.3 使用 mapper.py 完成 PTQ 与编译

执行位置：OE 容器。

`mapper.py` 是 Model Zoo 提供的转换入口，会串联预处理配置、PTQ 和编译。PTQ（训练后量化）用校准图估计数值范围，无需重新训练模型。日志中首次出现的 `hb_compile` 是 OE 编译阶段，它把量化模型编译为 S100 可执行的 HBM，无需单独运行。

```bash
# 进入转换工作目录。
cd /data/conversion
# 以 nash-e 为目标，使用 16 个并行任务和 O2 优化完成 PTQ 与编译。
python3 /data/model_zoo_s/samples/Vision/ultralytics_yolo/x86/mapper.py \
  --onnx /data/exchange/onnx/yolov5nu_coco2017_val128_s100.onnx \
  --cal-images /data/conversion/cal_images \
  --output-dir /data/conversion/output \
  --march nash-e --jobs 16 --optimize-level O2 --save-cache True

# 找到 mapper 生成的 NV12 HBM，并统一为对外交付名。
HBM_SOURCE="$(find /data/conversion/output -maxdepth 1 -type f \
  -name '*_nashe_640x640_nv12.hbm' | sort | head -1)"
mv "$HBM_SOURCE" /data/conversion/output/yolov5nu_s100_nv12.hbm
# 列出编译产物和大小。
ls -lh /data/conversion/output/
```

编译成功时，日志包含 `The hb_compile completes running.`，且最终模型为 `/data/conversion/output/yolov5nu_s100_nv12.hbm`。

<!-- IMAGE_SLOT:S2-01 -->
> **待补图 S2-01｜OE 编译完成**
> 建议文件：`docs/images/S2-01-compile-complete.png`
> 截图内容：OE 编译完成终端。
> 图注：OpenExplorer 完成 PTQ 与 HBM 编译。

### 2.4 查看量化结果

执行位置：OE 容器。

`conversion/output/hb_compile.log` 记录六路输出的 calibrated/quantized cosine。cosine 越接近 1，表示量化前后对应输出越接近；它是输出张量诊断，不等于数据集精度。本项目实测值如下：

| 输出 | Calibrated Cosine | Quantized Cosine |
|---|---:|---:|
| `output0` | 0.999866 | 0.986857 |
| `371` | 0.995730 | 0.993678 |
| `379` | 0.999687 | 0.999504 |
| `387` | 0.996379 | 0.994489 |
| `395` | 0.999411 | 0.999124 |
| `403` | 0.995772 | 0.993587 |

```bash
# 快速定位六路输出的量化诊断行。
grep -E 'output0|371|379|387|395|403' /data/conversion/output/hb_compile.log
```

量化阶段的主要产物是 `conversion/output/hb_compile.log` 和 `conversion/output/yolov5nu_s100_nv12.hbm`。

<!-- IMAGE_SLOT:S2-02 -->
> **待补图 S2-02｜六路量化 cosine**
> 建议文件：`docs/images/S2-02-six-output-cosine.png`
> 截图内容：六路 calibrated/quantized cosine。
> 图注：六路输出的量化前后余弦相似度。

### 2.5 RoboGo 量化

RoboGo 量化将在后续补充截图和经验证的平台流程。

## Stage 3：S100 板端部署

### 3.1 配置板端部署环境

执行位置：先在 OE 容器执行 `exit` 回到开发主机终端，再通过 SSH 在 S100 安装依赖。S100 应已启动配套系统，具备 BPU 运行时和模型运行测试工具 `hrt_model_exec`，并与主机网络互通。将 `<S100_IP>` 替换为板卡实际 IP。

```bash
export S100_HOST="root@<S100_IP>"
export BOARD_ROOT="/root/s100_yolov5u_train2deploy"
# 在 S100 安装一次推理与 COCO 评估所需依赖。
ssh "$S100_HOST" "python3 -m pip install numpy pycocotools"
```

### 3.2 上传部署文件

执行位置：开发主机。复用 Stage 1 的 `PROJECT_ROOT`、`DATASET_ROOT`、`MODEL_ZOO_ROOT`。

```bash
ssh "$S100_HOST" "mkdir -p '$BOARD_ROOT/output'"
scp -r "$MODEL_ZOO_ROOT/samples/Vision/ultralytics_yolo/py" \
  "$S100_HOST:$BOARD_ROOT/"
scp "$PROJECT_ROOT/conversion/output/yolov5nu_s100_nv12.hbm" \
  "$PROJECT_ROOT/scripts/run_model_zoo_bpu_subset.py" \
  "$PROJECT_ROOT/scripts/evaluate_coco_official.py" \
  "$PROJECT_ROOT/scripts/run_with_bpu_monitor.sh" \
  "$DATASET_ROOT/annotations/instances_test.json" \
  "$S100_HOST:$BOARD_ROOT/"
scp -r "$DATASET_ROOT/images/test" "$S100_HOST:$BOARD_ROOT/"
# 登录后，后续 3.3–3.6 的命令均在 S100 执行。
ssh "$S100_HOST"
```

板端项目目录现在包含 `py/`、HBM、三个辅助脚本、`instances_test.json`、16 张测试图所在的 `test/` 和 `output/`。本阶段的检测图、评估 JSON 和性能日志全部保留在 S100 的 `output/`，直接在板端查看。

### 3.3 查看模型信息

执行位置：S100。`hrt_model_exec` 是板端模型运行与测试工具；`model_info` 用于读取 HBM 的输入输出信息。

```bash
cd /root/s100_yolov5u_train2deploy
hrt_model_exec model_info --model_file ./yolov5nu_s100_nv12.hbm \
  | tee ./output/model_info.log
```

检查输入为 NV12 的 `images_y`、`images_uv`，对应 640×640 图像，输出与 1.5 的六路 cls/bbox 一致。终端内容同时写入 `output/model_info.log`。

<!-- IMAGE_SLOT:S3-01 -->
> **待补图 S3-01｜板端模型信息**
> 建议文件：`docs/images/S3-01-model-info.png`
> 截图内容：S100 终端的 NV12 双输入与六路输出信息。
> 图注：HBM 的输入输出与导出、编译配置一致。

### 3.4 单张图片检测

执行位置：S100 / `/root/s100_yolov5u_train2deploy`。

```bash
python3 -m py.rdk_yolo_app \
  --model-path ./yolov5nu_s100_nv12.hbm \
  --source ./test/000000000632.jpg \
  --workspace ./output/single_image \
  --mode default --yolo-type yolov5u --model-type detect \
  --score-thres 0.25 --nms-thres 0.7
```

Model Zoo 完成图片预处理、BPU 推理、解码和 NMS，检测图保存为 `output/single_image/000000000632.jpg_result.jpg`。在板端图像查看器中打开，检查框的位置与类别；单图用于直观核对，精度以接下来的固定测试集评估为准。

<!-- IMAGE_SLOT:S3-02 -->
> **待补图 S3-02｜单图检测结果**
> 建议文件：`docs/images/S3-02-detection-result.jpg`
> 截图内容：板端输出的检测图及目标框、类别与分数。
> 图注：S100 上的 YOLOv5nu 单图检测结果。

### 3.5 运行板端精度评估

执行位置：S100 / `/root/s100_yolov5u_train2deploy`。使用同一组 16 张测试图、score 0.25 和 NMS IoU 0.7；子集脚本复用 Model Zoo 检测器，监控脚本同步记录 Python 应用输出和 BPU 采样。

```bash
bash ./run_with_bpu_monitor.sh \
  --log ./output/bpu_inference.log \
  --interval 0.2 \
  -- \
  python3 ./run_model_zoo_bpu_subset.py \
    --model-zoo-yolo-dir ./py \
    --model-path ./yolov5nu_s100_nv12.hbm \
    --source ./test \
    --output ./output/bpu_result.txt \
    --score-thres 0.25 \
    --nms-thres 0.7
```

终端应显示 `images=16`，检测记录写入 `output/bpu_result.txt`。`output/bpu_inference.log` 同时包含带时间戳的 `[BPU] ratio=…%` 和结尾的 `[BPU_SUMMARY] samples=… average=… peak=…`；具体数值以本次板端实测为准。

脚本优先读取 `/sys/devices/system/bpu/bpu0/ratio`，不可读时使用 `/sys/devices/system/bpu/ratio`。这是板级 BPU 利用率的采样值，其他 BPU 任务也会影响它。0.2 秒的采样间隔可能错过约 1.2 ms 的单次 BPU 任务，因此这里观察整个 16 图循环，不把某一次采样当作单图是否使用 BPU 的证明。汇总是整个 Python 应用运行期间采样值的均值与峰值，包含初始化、读图和前后处理期间；它不能换算为单图推理延迟。

<!-- IMAGE_SLOT:S3-05 -->
> **待补图 S3-05｜BPU 利用率日志**
> 建议文件：`docs/images/S3-05-bpu-utilization-log.png`
> 截图内容：同一次 16 图运行中的 `[BPU]` 采样、`images=16` 和 `[BPU_SUMMARY]`。
> 图注：Python 推理应用运行期间的板级 BPU 采样；数值待板端实测补入。

继续在 S100 将检测记录转换为 COCO 预测 JSON，再调用未修改的 `pycocotools` 评估：

```bash
python3 ./evaluate_coco_official.py bpu-to-json \
  --bpu-text ./output/bpu_result.txt \
  --annotations ./instances_test.json \
  --output ./output/quantization_bpu_predictions.json
python3 ./evaluate_coco_official.py evaluate \
  --annotations ./instances_test.json \
  --predictions ./output/quantization_bpu_predictions.json \
  --metrics ./output/quantization_bpu_metrics.json
```

终端打印 COCO 指标和 JSON 汇总，`image_count` 应为 16；预测与指标均保存在板端 `output/`。mAP50-90 表示 IoU 0.50–0.90、步长 0.05 的 AP 均值。本项目已有参考实测记录见[精度评估](精度评估.md)：

| 指标 | FP32 | S100 BPU |
|---|---:|---:|
| mAP50 | 0.388 | 0.414 |
| mAP50-90 | 0.292 | 0.321 |
| mAP50-95 | 0.266 | 0.293 |

这里只使用 16 张图，抽样波动较大；量化扰动可能使阈值附近的候选框通过或落出筛选，Ultralytics 与 Model Zoo 的预处理、后处理差异也可能影响结果。因此本次 BPU 指标略高不能说明量化通常提升精度；重新训练和评估时，以当次输出为准。

<!-- IMAGE_SLOT:S3-03 -->
> **待补图 S3-03｜BPU 精度结果**
> 建议文件：`docs/images/S3-03-bpu-metrics.png`
> 截图内容：板端三项 mAP 与 `image_count=16` 的评估汇总。
> 图注：与 FP32 使用相同测试图和阈值的 S100 BPU 精度评估。

### 3.6 测试 BPU 性能

执行位置：S100 / `/root/s100_yolov5u_train2deploy`。

```bash
hrt_model_exec perf --thread_num 1 --model_file ./yolov5nu_s100_nv12.hbm \
  | tee ./output/perf_1thread.log
hrt_model_exec perf --thread_num 2 --model_file ./yolov5nu_s100_nv12.hbm \
  | tee ./output/perf_2thread.log
```

两次结果直接显示在板端终端，并分别保存到 `output/perf_1thread.log` 和 `output/perf_2thread.log`。已有参考记录每项 200 帧，见[延迟评估](延迟评估.md)：

| 线程数 | 平均 BPU 任务延迟 | 吞吐 |
|---:|---:|---:|
| 1 | 1.233 ms | 794.452 FPS |
| 2 | 1.867 ms / 线程 | 1053.081 FPS |

`perf` 测量提交 BPU 任务到等待完成的运行时性能，不含读图、NV12 构造和 Python 后处理，不能视为完整业务的端到端延迟。两线程增加并发吞吐，单任务延迟也可能上升。3.5 的 `bpu_inference.log` 覆盖完整 Python 应用期间的日志与利用率采样；这里的任务延迟、吞吐和前面的利用率是不同指标。

<!-- IMAGE_SLOT:S3-04 -->
> **待补图 S3-04｜BPU 性能测试**
> 建议文件：`docs/images/S3-04-bpu-performance.png`
> 截图内容：板端一线程、两线程 perf 的延迟与吞吐结果。
> 图注：HBM 的 BPU 任务性能，不包含 Python 应用前后处理。

## Stage 4：结果总结

| 阶段 | 执行位置 | 输入 | 输出 |
|---|---|---|---|
| 训练与 FP32 基线 | 开发主机 / Conda | 预训练权重、固定数据集 | `train/runs/coco2017_val128_s100/weights/best.pt`、`validation/quantization_fp32_metrics.json` |
| 导出 | 开发主机 / Conda | `best.pt`、Model Zoo 导出脚本 | `exchange/onnx/yolov5nu_coco2017_val128_s100.onnx`，六路输出 |
| 量化与编译 | OE 容器，项目挂载到 `/data` | 六路 ONNX、20 张校准图 | `conversion/output/yolov5nu_s100_nv12.hbm`、`hb_compile.log` |
| 部署、评估与性能测试 | S100 / `/root/s100_yolov5u_train2deploy` | HBM、Model Zoo `py/`、辅助脚本、16 张测试图及标注 | 板端 `output/` 下的检测图、`model_info.log`、`bpu_result.txt`、`quantization_bpu_predictions.json`、`quantization_bpu_metrics.json`、`bpu_inference.log`、`perf_1thread.log`、`perf_2thread.log` |

## QA：常见问题

| 问题 | 排查与处理 |
|---|---|
| OE 下载中断或解压失败 | 使用 0.2 的 `wget -c` 续传；核对两个文件名和 3.7.0 版本，下载完成后重新解压。 |
| `docker load` 失败 | 确认 Docker 服务已启动、当前用户有权限，镜像 tar 下载完整且磁盘空间足够；不要把工具链 tgz 当成镜像加载。 |
| 容器中找不到 `/data` 下的文件 | 返回主机检查传给 `run_docker.sh` 的 `PROJECT_ROOT` 是否为本次克隆目录，并核对该目录下确有 ONNX 和校准图。 |
| ONNX 不是六路输出 | 确认 1.5 使用 Model Zoo 的 `export_monkey_patch.py`，核对三组 cls/bbox 名称与形状，再重新量化；通用导出结果不能直接替代。 |
| 校准图不是 20 张 | 核对训练集目录及 `.jpg` 扩展名；少于 20 张时检查数据完整性，多于 20 张时移出旧校准图，再按 2.1 的排序规则准备。 |
| 未生成 HBM，或重命名时报源文件为空 | 从 `conversion/output/hb_compile.log` 查找首个编译错误，确认 ONNX、校准路径与 `nash-e` 配置；先取得编译成功日志及 NV12 HBM，再执行重命名。 |
| SSH 或上传失败 | 核对 `S100_HOST` 中的 IP、主机与板卡网络、板端 SSH 服务及登录凭据；确认 `BOARD_ROOT` 可写，所有上传命令在主机执行。 |
| 板端模块缺失或工具不可用 | 确认使用 S100 配套系统的 Python 与 BPU 运行时；`numpy`、`pycocotools` 在 3.1 安装。Model Zoo 入口会检查并尝试安装 `tqdm`、`scipy`、`numpy`、OpenCV，首次运行需能访问包源。若 `hrt_model_exec` 或 BPU 运行时缺失，应检查板端系统环境。 |
| 检测框位置或类别明显不对 | 核对 HBM 的 NV12 输入、六路输出及 640×640 尺寸，保持 `yolov5u`、`detect` 和同一 Model Zoo 预处理/后处理；确认模型为本次训练编译产物。 |
| FP32 与 BPU 的 mAP 不同 | 先核对同一组 16 张图与标注、score 0.25、NMS IoU 0.7、类别映射和同一权重链路；再考虑小样本波动、阈值附近量化变化及两套预处理/后处理差异。不要将本例数值推广为完整 COCO 精度结论。 |
