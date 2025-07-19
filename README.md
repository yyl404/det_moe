# 基于多Head预测器的置信度排序融合方法

这是一个基于MMYOLO, MMDetection框架所构建的，用于演示基于多Head预测器的置信度排序融合方法的项目。

## 安装

```bash
# 安装依赖
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install "mmcv>=2.0.0rc4,mmcv<2.1.0"
mim install mmdet==3.3.0
mim install mmenegine
pip install albumentations
pip install 'numpy<2' # 使用高于2.0版本的numpy会导致错误

# 安装自定义包
cd det_moe
pip install -e . -v
```

## 使用方法

### 训练
```bash
python tools/train.py configs/gru/yolo_gru_s_syncbn_fast_8xb16-500e_voc.py
```

### 测试
```bash
python tools/test.py configs/gru/yolo_gru_s_syncbn_fast_8xb16-500e_voc.py work_dirs/your_model/latest.pth
```

### 推理
```bash
python tools/inference.py configs/gru/yolo_gru_s_syncbn_fast_8xb16-500e_voc.py work_dirs/your_model/latest.pth --img path/to/image.jpg
```

## 实验结果

|Models|val mAP[50]|val mAP[50,95]|test mAP[50]|test mAP[50,95]|
|---|---|---|
|yolov8s|||
|yolov7s|||
|yolov6s|||
|yolov5s|||
|yolo-gru|71.0|52.9|72.0|51.9|


# 部署至晟腾NPU

目前在晟腾NPU上支持对模型的测试和推理，部署方法如下