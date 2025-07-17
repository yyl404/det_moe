# Copyright (c) OpenMMLab. All rights reserved.
# Modified by YYL
import os
from argparse import ArgumentParser
from pathlib import Path

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils import ProgressBar, path

import det_moe
from det_moe.registry import VISUALIZERS


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--deploy',
        action='store_true',
        help='Switch model to deployment mode')
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Whether to use test time augmentation')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument(
        '--class-name',
        nargs='+',
        type=str,
        help='Only Save those classes if set')
    args = parser.parse_args()
    return args


def get_file_list(img_path):
    """Get file list from path."""
    if os.path.isfile(img_path):
        files = [img_path]
        source_type = {'is_dir': False}
    elif os.path.isdir(img_path):
        files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            files.extend(list(Path(img_path).glob(ext)))
        source_type = {'is_dir': True}
    else:
        raise FileNotFoundError(f'Cannot find {img_path}')
    
    return files, source_type


def main():
    args = parse_args()

    config = args.config

    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    
    if 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None

    if args.tta:
        assert 'tta_model' in config, 'Cannot find ``tta_model`` in config.' \
            " Can't use tta !"
        assert 'tta_pipeline' in config, 'Cannot find ``tta_pipeline`` ' \
            "in config. Can't use tta !"
        config.model = ConfigDict(**config.tta_model, module=config.model)
        test_data_cfg = config.test_dataloader.dataset
        while 'dataset' in test_data_cfg:
            test_data_cfg = test_data_cfg['dataset']

        # batch_shapes_cfg will force control the size of the output image,
        # it is not compatible with tta.
        if 'batch_shapes_cfg' in test_data_cfg:
            test_data_cfg.batch_shapes_cfg = None
        test_data_cfg.pipeline = config.tta_pipeline

    # build the model from a config file and a checkpoint file
    model = init_detector(
        config, args.checkpoint, device=args.device, cfg_options={})

    if args.deploy:
        # Switch to deploy mode if needed
        if hasattr(model, 'switch_to_deploy'):
            model.switch_to_deploy()

    if not args.show:
        path.mkdir_or_exist(args.out_dir)

    # init visualizer
    if hasattr(config, 'visualizer'):
        visualizer = VISUALIZERS.build(config.visualizer)
        visualizer.dataset_meta = model.dataset_meta
    else:
        visualizer = None

    # get file list
    files, source_type = get_file_list(args.img)

    # get model class name
    dataset_classes = model.dataset_meta.get('classes', [])

    # check class name
    if args.class_name is not None:
        for class_name in args.class_name:
            if class_name in dataset_classes:
                continue
            print(f'Available classes: {dataset_classes}')
            raise RuntimeError(
                'Expected args.class_name to be one of the list, '
                f'but got "{class_name}"')

    # start detector inference
    progress_bar = ProgressBar(len(files))
    for file in files:
        result = inference_detector(model, file)

        img = mmcv.imread(file)
        img = mmcv.imconvert(img, 'bgr', 'rgb')

        if source_type['is_dir']:
            filename = os.path.relpath(file, args.img).replace('/', '_')
        else:
            filename = os.path.basename(file)
        out_file = None if args.show else os.path.join(args.out_dir, filename)

        progress_bar.update()

        # Get candidate predict info with score threshold
        pred_instances = result.pred_instances[
            result.pred_instances.scores > args.score_thr]

        if visualizer is not None:
            visualizer.add_datasample(
                filename,
                img,
                data_sample=result,
                draw_gt=False,
                show=args.show,
                wait_time=0,
                out_file=out_file,
                pred_score_thr=args.score_thr)
        else:
            # Simple visualization if no visualizer
            if not args.show:
                mmcv.imwrite(img, out_file)

    if not args.show:
        print_log(
            f'\nResults have been saved at {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main() 