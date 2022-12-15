import warnings
from functools import partial
from itertools import count
from typing import Dict, List

import torch
import torch.nn as nn
from tqdm import tqdm

from configs import args
from dataset import get_dataset
from model import FullModel, get_learner
from utils.metric import Metric as Metric
from utils.utils import loss_with_ignored_area, update_metric


def run_one_batch(model:FullModel, batch, iou_metric:Metric, loss_func, is_train):
    outer_loss = []
    
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

            loss_ = loss_func(logits, query_gt.float(), ignore_area)
            
            loss = loss_ / num_tasks_this_batch
            outer_loss.append(loss_.detach())

        if is_train:
            loss.backward()

        update_metric(logits, cls_name, cls_index, query_gt, ignore_area, iou_metric)
    
    return sum(outer_loss)/len(outer_loss)


def test(model, test_dataloader, optimizers, loss_func, global_step):
    test_iou_metric = Metric()
    
    model.eval()
    for idx, batch in enumerate(tqdm(test_dataloader, desc=f'test')):
        run_one_batch(model, batch, test_iou_metric, loss_func, is_train=False)
        
    test_miou, test_fbiou = test_iou_metric.compute_iou()
    
    print(f'[test] tot_batch:{idx} miou:{test_miou:.3f} fbiou:{test_fbiou:.3f}')
    

def main():
    
    args.epochs = args.warmup_epochs + args.training_epochs
    
    loss_func = nn.BCEWithLogitsLoss(reduction='none')
    loss_func = partial(loss_with_ignored_area, loss_func)
    
    train_dataloader, test_dataloader = get_dataset()

    model, optimizers = get_learner(len(train_dataloader))
    
    test_freq = 3
    test_idx = len(train_dataloader) // test_freq
    
    global_step = 0
    for epoch in count(start=0, step=1):
        if epoch >= args.epochs:
            warnings.warn(f"\n\nEpoch({epoch}) is out of range of args.epoch({args.epochs}). This may affect schedulers.\n\n")
        
        for idx, batch in enumerate(train_dataloader):
            train_iou_metric = Metric()
            
            model.train()
            optimizers.schedule(['fusion'], global_step)
            optimizers.zero_grad()
            outer_loss = run_one_batch(model, batch, train_iou_metric, loss_func, is_train=True)
            optimizers.step()
            test_miou, test_fbiou = train_iou_metric.compute_iou()
            
            print(f'[train {epoch}] batch_idx:{idx} loss:{outer_loss:.3f} miou:{test_miou:.3f} fbiou:{test_fbiou:.3f}')
            global_step += 1

            if idx % test_idx == 0:
                test(model, test_dataloader, optimizers, loss_func, global_step)
                
if __name__ == "__main__":
    main()
