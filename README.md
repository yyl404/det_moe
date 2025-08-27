# 基于多Head预测器的置信度排序融合方法

这是一个基于MMYOLO, MMDetection框架所构建的，用于演示基于多Head预测器的置信度排序融合方法的项目。

## 安装

```bash
# 安装依赖
conda create -n mmlab python=3.10 -y
conda activate mmlab
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install mmenegine
mim install mmdet==3.3.0
mim install mmyolo==0.6.0
pip install albumentations
pip install 'numpy<2' # 使用高于2.0版本的numpy会导致错误

# 安装本项目的代码
git clone https://github.com/yyl404/det_moe.git
cd det_moe
pip install -e . -v
```

## 配置数据集

相关的实验结果是使用数据集Pascal VOC得到的，我们使用了按照coco标注格式重新整理的Pascal VOC

下载链接：https://pan.baidu.com/s/1_3FCy3W51NhQQ5ApiihMHA?pwd=xpz7

提取码: xpz7 

将数据集下载后，放置在路径det_moe/data/voc_as_coco下。

目录结构是：
```
|- det_moe
    |- data
        |- voc_as_coco
            |- annotations
                |- instances_test.json
                |- instances_train.json
                |- instances_val.json
            |- images
                |- test
                |- train
                |- val
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

|Models|test mAP[50,95]|test mAP[50]|
|---|---|---|
|yolov8s|46.10|66.80|
|yolov7s|-|-|
|yolov6s|43.70|66.20|
|yolov5s|38.00|62.70|
|yolo-gru|-|-|

# 部署至晟腾NPU

目前在晟腾NPU上支持对模型的测试，推理和训练，部署方法参考：[昇腾NPU环境配置指南](docs/npu_setup.md)