import os
import pickle
from pathlib import Path

import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset

from configs import args
from utils.utils import aug_data

from .resize_crop import RandomCropResize, Resize


class DatasetCOCO(Dataset):
    def __init__(self, datapath, fold, random_transform, split):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.benchmark = 'coco'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        
        if self.split == 'trn':
            self.img_path = Path(datapath) / "train2014"
            self.gt_path = Path(datapath) / "annotation" / "train2014"
        else:
            self.img_path = Path(datapath) / "val2014"
            self.gt_path = Path(datapath) / "annotation" / "val2014"
        
        self.rand_transform = random_transform
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        with open("./dataset/splits/coco/labels.txt") as f:
            self.all_class_name = f.readlines()
        self.all_class_name = [t.strip() for t in self.all_class_name]
        if split == 'trn':
            self.crop_resize_trans = RandomCropResize(args.target_size, args.scale_range)
        else:
            def identity(*args):
                return args
            if args.test_with_org_resolution:
                self.crop_resize_trans = identity
            else:
                self.crop_resize_trans = Resize(args.target_size)

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        image_name, class_sample = self.sample_episode(idx)
        images, masks, image_size = self.load_frame(image_name)

        images = [t.to(args.device) for t in images]
        masks = [t.to(args.device) for t in masks]
        
        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)

        batch = {
                'query_img': images,
                'query_mask': masks[:,0],
                'query_name': image_name,
                'query_idx': idx,

                'org_query_imsize': image_size,

                'class_id': torch.tensor(class_sample+1),
                'class_name': self.all_class_name[class_sample+1]
                }

        return batch

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val

        return class_ids

    def build_img_metadata_classwise(self):
        with open('dataset/splits/coco/%s/fold%d.pkl' % (self.split, self.fold), 'rb') as f:
            img_metadata_classwise = pickle.load(f)
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += [(n, k) for n in self.img_metadata_classwise[k]]
        img_metadata = sorted(list(set(img_metadata)))
        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))
        return img_metadata

    def read_mask(self, name):
        mask_path = os.path.join(self.img_path, 'annotations', name)
        mask = torch.tensor(np.array(Image.open(mask_path[:mask_path.index('.jpg')] + '.png')))
        return mask
    
    def sample_episode(self, idx):
        image_name, class_sample = self.img_metadata[idx]
        return image_name, class_sample
    
    def load_frame(self, image_name):
        need_trans = True
        image, mask, image_size = self.data_request(image_name, return_org_size=True, need_trans=need_trans)

        aug_times = 0 if self.split=='val' else 1
        images, masks = aug_data([image], [mask], aug_times, self.rand_transform)
        return images, masks, image_size

    def data_request(self, img_name, return_org_size=False, need_trans=True):
        img = np.array(Image.open(self.img_path / (img_name)).convert('RGB'))
        mask = np.array(Image.open(self.gt_path / (img_name.replace(".jpg", ".png"))))[...,None]
        org_size = img.shape
        if need_trans:
            img, mask = self.crop_resize_trans(img, mask)
        else:
            img, mask = img, mask
        if return_org_size:
            return img, mask, org_size
        else:
            return img, mask
