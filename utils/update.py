import yaml
from yaml.loader import SafeLoader

# {'classes': ['bz', 'chem', 'dis', 'hz', 'ul', 'yd'], 
#  'data_root': './fst_coco/', 
#  'train_ann_file': 'annotations/train.json', 
#  'train_data_prefix': 'images/train/', 
#  'val_ann_file': 'annotations/val.json', 
#  'val_data_prefix': 'images/val/'}

def update_data_cfg(data_info_file):
    options = {}

    with open(data_info_file) as f:
        data = yaml.load(f, Loader=SafeLoader)
        
    options['classes'] = tuple(data['classes'])
    options['train_dataloader.dataset.data_root'] = data['data_root']
    options['train_dataloader.dataset.metainfo'] = dict(classes=tuple(data['classes']))
    options['train_dataloader.dataset.ann_file'] = data['train_ann_file']
    options['train_dataloader.dataset.data_prefix'] = dict(img=data['train_data_prefix'])
    
    options['val_dataloader.dataset.data_root'] = data['data_root']
    options['val_dataloader.dataset.metainfo'] = dict(classes=tuple(data['classes']))
    options['val_dataloader.dataset.ann_file'] = data['val_ann_file']
    options['val_dataloader.dataset.data_prefix'] = dict(img=data['val_data_prefix'])
    
    options['val_evaluator.ann_file'] = data['data_root'] + data['val_ann_file']
  
    return options
    
    
def update_after_merge(cfg):
    cfg['test_dataloader'] = cfg['val_dataloader']
    cfg['test_evaluator'] = cfg['val_evaluator']
    
    if 'bbox_head' in cfg['model']:
        if isinstance(cfg['model']['bbox_head'], list):
            for head in cfg['model']['bbox_head']:
                head['num_classes'] = len(cfg['classes'])
        else: cfg['model']['bbox_head']['num_classes'] = len(cfg['classes'])
    else :
        if isinstance(cfg['model']['roi_head']['bbox_head'], list):
            for head in cfg['model']['roi_head']['bbox_head']:
                head['num_classes'] = len(cfg['classes'])
        else: cfg['model']['roi_head']['bbox_head']['num_classes'] = len(cfg['classes'])
    
    return cfg