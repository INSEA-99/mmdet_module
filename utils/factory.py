from mmengine.config import Config, DictAction
from mmengine.runner import Runner
import os.path as osp

from utils import update_data_cfg, update_after_merge


def create_model(**kwrags):
    config = make_config(kwrags)
    runner = Runner.from_cfg(config)
    return runner

def make_config(kwrags):
    init_cfg = Config.fromfile(kwrags['config'])
    # overwrite_cfg = overwrite_config(init_cfg, **kwargs)
    merged_cfg = merge_args(init_cfg, kwrags['data_yaml'])
    # merged_cfg = merge_args(overwrite_cfg, args)
    if 'preprocess_cfg' in merged_cfg:
        merged_cfg.model.setdefault('data_preprocessor',
                             merged_cfg.get('preprocess_cfg', {}))
    return merged_cfg

# def overwrite_config(cfg):
#     cfg['classes'] = tuple()

#     options = update_data_cfg(cfg, args.data_info_file)
#     cfg.merge_from_dict(options, False)
#     cfg = update_after_merge(cfg)
#     return

def merge_args(cfg, data_yaml):
    cfg['work_dir'] = './results'
    cfg['classes'] = tuple()

    options = update_data_cfg(data_yaml)
    cfg.merge_from_dict(options, False)
    cfg = update_after_merge(cfg)

    return cfg
