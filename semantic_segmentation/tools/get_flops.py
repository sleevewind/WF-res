import argparse
import os
import sys
from copy import deepcopy

import torch
from mmcv import Config, DictAction

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
SEG_ROOT = os.path.abspath(os.path.join(CUR_DIR, '..'))
if SEG_ROOT not in sys.path:
    sys.path.insert(0, SEG_ROOT)

from mmseg.models import build_segmentor

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError as exc:
    raise ImportError('Please upgrade mmcv to >0.6.2') from exc

# Register local custom backbones.

from backbone import wfres  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description='Compute FLOPs/Params for a segmentor')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[2048, 512],
        help='input image size, e.g. --shape 2048 512')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in the config, the key-value pair in xxx=yyy '
        'format will be merged into the config file')
    return parser.parse_args()


def _replace_syncbn(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == 'norm_cfg' and isinstance(value, dict) and value.get('type') == 'SyncBN':
                value['type'] = 'BN'
            else:
                _replace_syncbn(value)
    elif isinstance(obj, list):
        for item in obj:
            _replace_syncbn(item)


def _prepare_config(cfg):
    cfg = deepcopy(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    _replace_syncbn(cfg.model)
    return cfg


def main():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    cfg = _prepare_config(cfg)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            f'FLOPs counter is not supported with {model.__class__.__name__}')

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(
        f'{split_line}\n'
        f'Config: {args.config}\n'
        f'Input shape: {input_shape}\n'
        f'Flops: {flops}\n'
        f'Params: {params}\n'
        f'{split_line}'
    )
    print(
        'Please double-check whether all custom ops are counted correctly '
        'before using the number in a paper.'
    )


if __name__ == '__main__':
    main()
