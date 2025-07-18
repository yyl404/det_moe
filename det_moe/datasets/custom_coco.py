# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path
from mmdet.datasets.base_det_dataset import BaseDetDataset

from det_moe.registry import DATASETS
from .api_wrappers import COCO


@DATASETS.register_module()
class CustomCocoDataset(BaseDetDataset):
    """Automatic COCO format dataset class that reads class information from annotation files.
    
    This class inherits from BaseDetDataset and can handle COCO format annotation files,
    automatically reading class information from annotation files and generating corresponding
    color palettes without manual setting of classes and colors, suitable for different
    object detection tasks.
    
    Args:
        classes (tuple, optional): Manually specified class name tuple, if None then automatically read from annotation file
        palette (list, optional): Manually specified color list, if None then automatically generated
        auto_load_classes (bool): Whether to automatically load class information from annotation file, default True
        **kwargs: Other parameters passed to parent class
    """

    def __init__(self, 
                 classes: tuple = None,
                 palette: List[tuple] = None,
                 auto_load_classes: bool = True,
                 **kwargs) -> None:
        self.auto_load_classes = auto_load_classes
        self._manual_classes = classes
        self._manual_palette = palette
        
        # If auto-loading is enabled, set empty metainfo first, update in load_data_list
        if auto_load_classes:
            self._metainfo = {
                'classes': classes or (),
                'palette': palette or []
            }
        else:
            # Manual mode, use provided classes and palette
            if classes is None:
                raise ValueError("When auto_load_classes=False, classes parameter must be provided")
            self._metainfo = {
                'classes': classes,
                'palette': palette or self._generate_default_palette(len(classes))
            }
        
        super().__init__(**kwargs)

    def _load_classes_from_annotation(self) -> None:
        """Automatically read class information from annotation file.
        
        This method reads categories information from COCO format annotation file,
        extracts class names and generates corresponding color palette.
        """
        # Get all category information
        categories = self.coco.load_cats(self.coco.get_cat_ids())
        
        # Sort by category_id to ensure consistent class order
        categories.sort(key=lambda x: x['id'])
        
        # Extract class names
        classes = tuple(cat['name'] for cat in categories)
        
        # If manually specified classes, check if they match
        if self._manual_classes is not None:
            if set(classes) != set(self._manual_classes):
                print(f"Warning: Classes in annotation file {classes} do not match manually specified classes {self._manual_classes}")
                print("Using class information from annotation file")
        
        # Generate color palette
        palette = self._manual_palette or self._generate_default_palette(len(classes))
        
        # Update metainfo
        self._metainfo.update({
            'classes': classes,
            'palette': palette
        })
        
        print(f"Auto-loaded class information: {classes}")
        print(f"Number of classes: {len(classes)}")

    def _generate_default_palette(self, num_classes: int) -> List[tuple]:
        """Generate default color palette.
        
        Args:
            num_classes (int): Number of classes
            
        Returns:
            List[tuple]: Color list
        """
        import colorsys
        palette = []
        for i in range(num_classes):
            hue = i / num_classes
            saturation = 0.8
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            palette.append(tuple(int(x * 255) for x in rgb))
        return palette

    def load_data_list(self) -> List[dict]:
        """Load COCO format annotation file and automatically read class information.
        
        Returns:
            List[dict]: Parsed data list
        """
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = COCO(local_path)
        
        # Automatically read class information from annotation file
        if self.auto_load_classes:
            self._load_classes_from_annotation()
        
        # Get category ID mapping
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info': raw_ann_info,
                'raw_img_info': raw_img_info
            })
            data_list.append(parsed_data_info)
        
        # Verify uniqueness of annotation IDs
        assert len(set(total_ann_ids)) == len(total_ann_ids), \
            f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation data to target format.
        
        Args:
            raw_data_info (dict): Raw data information loaded from annotation file
            
        Returns:
            Union[dict, List[dict]]: Parsed annotation data
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # Build image path
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
            
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        # If class information needs to be returned (for open vocabulary detection, etc.)
        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        # Parse instance annotations
        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            # Skip ignored annotations
            if ann.get('ignore', False):
                continue
                
            # Parse bounding box
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            # Set ignore flag
            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
                
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            # If segmentation annotation exists
            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
            
        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotation data according to filter configuration.
        
        Returns:
            List[dict]: Filtered data list
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # Get image IDs with annotations
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # Get image IDs with required category annotations
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # Combine two conditions
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ID list for image at specified index.
        
        Args:
            idx (int): Data index
            
        Returns:
            List[int]: Category IDs of all instances in the image
        """
        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances] 