import torch
from torch.utils.data import Dataset

from configs import args

from .coco import DatasetCOCO
from .pascal import DatasetPASCAL


class CustomDataset(Dataset):
    def __init__(self,
        split,
        rand_aug_transform=None,
        ) -> None:
        dataset_class = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
        }
        split = 'trn' if split == 'train' else 'val'
        fold = 0 if args.fold is None else args.fold
        self.dataset = dataset_class[args.dataset_name](args.dataset_root, fold=fold, random_transform=rand_aug_transform, split=split)
        self.num_classes_per_task = 1

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        cur_class_name = sample['class_name']
        
        cls_name_selected = [cur_class_name]
        
        valid_idx = {name:self.dataset.all_class_name.index(name) for name in cls_name_selected}
        
        sample['input_text'] = cls_name_selected
        
        sample['input_cls_id'] = valid_idx 

        if self.dataset.benchmark in ['pascal', 'coco']:
            for key in ['query_mask']:
                gt = sample[key]
                n, h, w = gt.shape
                new_gt = torch.zeros((n, self.num_classes_per_task, h, w), device=gt.device)
                for i, cls_name in enumerate(cls_name_selected):
                    new_gt[:,i] = (gt==valid_idx[cls_name])
                ignore_mask = (gt==255)
                sample[key] = new_gt
                sample[f"{key}_ignore"] = ignore_mask

        return sample

def collect_fn(batch):
    batched_data = {}
    for sample in batch:
        if sample is None:
            continue
        for key, val in sample.items():
            lst = batched_data.get(key, [])
            lst.append(val)
            batched_data[key] = lst

    for key, val in batched_data.items():
        if isinstance(val[0], torch.Tensor):
            val = torch.stack(val, dim=0)
            batched_data[key] = val

    return batched_data
