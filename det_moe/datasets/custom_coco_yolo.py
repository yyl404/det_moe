from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset

from .custom_coco import CustomCocoDataset
from det_moe.registry import DATASETS

@DATASETS.register_module()
class CustomCocoYoloDataset(BatchShapePolicyDataset, CustomCocoDataset):
    """Custom COCO format YOLO dataset with batch shape policy support.
    
    This class combines the batch processing optimization functionality of
    BatchShapePolicyDataset with the automatic class loading functionality of
    CustomCocoDataset.
    """
    pass