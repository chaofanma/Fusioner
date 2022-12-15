from typing import Dict, List

import torch
from tqdm import tqdm

from configs import args
from dataset import get_dataset
from model import FullModel, get_learner
from utils.metric import Metric as Metric
from utils.utils import update_metric


def run_one_batch(model:FullModel, batch, iou_metric:Metric, is_train=False):
    
    query_imgs = batch['query_img']
    query_gts = batch['query_mask']
    cls_names:List[List[str]] = batch['input_text']
    cls_indexs:List[Dict[str,int]] = batch['input_cls_id']
    
    num_tasks_this_batch = len(cls_names)
    
    if 'query_mask_ignore' in batch.keys():
        ignore_areas = batch['query_mask_ignore']
    else:
        ignore_areas = [None] * num_tasks_this_batch

    for query_img, query_gt, cls_name, cls_index, ignore_area in zip(
        query_imgs, query_gts, cls_names, cls_indexs, ignore_areas):
        
        with torch.set_grad_enabled(is_train):
            if not is_train and args.test_with_org_resolution:
                logits = model.multi_scale_eval(query_img, cls_name)
            else:
                logits = model(query_img, cls_name)
        
        update_metric(logits, cls_name, cls_index, query_gt, ignore_area, iou_metric)
        

def test(model, test_dataloader):

    test_iou_metric = Metric()
    model.eval()

    for idx, batch in enumerate(tqdm(test_dataloader, desc=f'test')):
        run_one_batch(model, batch, test_iou_metric, is_train=False)
        
    test_miou, test_fbiou = test_iou_metric.compute_iou()
    
    print('test_miou', test_miou)
    print('test_fbiou', test_fbiou)
    
    return test_miou, test_iou_metric

def main():
    args.epochs = args.warmup_epochs + args.training_epochs
    
    train_dataloader, test_dataloader = get_dataset()
    model, _ = get_learner(len(train_dataloader))
    
    test(model, test_dataloader)
    
if __name__ == "__main__":
    main()