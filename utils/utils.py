from typing import Callable, List

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


def aug_data(imgs:torch.Tensor, labels:torch.Tensor, aug_times:int, augmenter:Callable):
    assert isinstance(aug_times, int)
    aug_imgs = []
    aug_labels = []
    if aug_times == 0:
        for img, label in zip(imgs, labels):
            ret_dict = augmenter(image=img, mask=label, do_aug=False)
            aug_imgs.append(ret_dict['image'])
            aug_labels.append(ret_dict['mask'])
    else:
        for _ in range(aug_times):
            for img, label in zip(imgs, labels):
                ret_dict = augmenter(image=img, mask=label, do_aug=True)
                aug_imgs.append(ret_dict['image'])
                aug_labels.append(ret_dict['mask'])
    return aug_imgs, aug_labels


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]



class Optimizers:
    def __init__(self, optimizers_dict: dict, schedules_dict: dict) -> None:
        self.optimizers_dict: dict = {k:v for k,v in optimizers_dict.items() if v is not None}
        self.schedules_dict = schedules_dict
        
    def step(self):
        for name, optimizer in self.optimizers_dict.items():
            optimizer.step()
            
    def zero_grad(self):
        for name, optimizer in self.optimizers_dict.items():
            optimizer.zero_grad()
        
    def schedule(self, names:List[str], global_step):
        wd_schedule = self.schedules_dict['wd']
        for name in names:
            lr_schedule = self.schedules_dict[name]
            for i, param_group in enumerate(self.optimizers_dict[name].param_groups):
                param_group["lr"] = lr_schedule[global_step if global_step < len(lr_schedule) else -1]
                if i == 0:
                    param_group["weight_decay"] = wd_schedule[global_step if global_step < len(wd_schedule) else -1]

    def state_dict(self):
        optim_state_dict, schedules_dict = {}, {}
        for name, optimizer in self.optimizers_dict.items():
            optim_state_dict[name] = optimizer.state_dict()
        for name, schedule in self.schedules_dict.items():
            schedules_dict[name] = schedule
        return (optim_state_dict, schedules_dict)
    
    def load(self, state_dict):
        for name, optimizer in state_dict[0].items():
            self.optimizers_dict[name].load_state_dict(optimizer)
        for name, schedule in state_dict[1].items():
            self.schedules_dict[name] = schedule


def get_preds(logits, thres=0.5):
    probs = torch.sigmoid(logits)
    b, c, h, w = probs.shape
    bg = torch.ones((b,1,h,w), device=logits.device)*thres
    probs_with_bg = torch.cat([bg, probs], dim=1)
    pred_label = probs_with_bg.argmax(dim=1)
    
    pred_label_onehot = F.one_hot(pred_label, c+1)
    pred_label_onehot = rearrange(pred_label_onehot[...,1:], 'b h w c -> b c h w')
    
    return pred_label_onehot


@torch.no_grad()
def update_metric(logits, input_text, cls_dict, gt, ignore_area, metric, pred_mask=None):
    if pred_mask is None:
        pred = get_preds(logits)
    else:
        pred = pred_mask
    pred_classwise = {key: torch.zeros_like(pred[:,0,...]) for key in input_text} 
    for idx_, key in enumerate(input_text):
        pred_classwise[key] += pred[:,idx_,...]
    for key, idx_ in cls_dict.items():
        obj_idx = input_text.index(key)
        pred_obj = pred_classwise[key]
        support_obj = gt[:, obj_idx, ...]
        if support_obj.sum() > 0:
            metric.update(pred_obj.int(), support_obj.int(), gt_ignore=ignore_area, class_id=[idx_])


def loss_with_ignored_area(loss_fn, input: torch.Tensor, target: torch.Tensor, ignore_area):
    loss = loss_fn(input, target)
    if (ignore_area is not None) and (ignore_area!=True).sum() > 0:
        loss = loss * (ignore_area != True)
        loss = loss.sum() / (ignore_area!=True).sum()
    else:
        loss = loss.mean()
        
    return loss
    
