import os
from pathlib import Path

import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset

from configs import args
from utils.utils import aug_data

from .resize_crop import RandomCropResize, Resize


class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, random_transform, split):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'pascal'

        self.img_path = Path(datapath) / 'VOCdevkit' / 'VOC2012/JPEGImages'
        self.ann_path = Path(datapath) / 'SegmentationClassAug'
        self.rand_transform = random_transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.all_class_name = [
            'background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
        self.img_cache = {}
        self.mask_cache = {}
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
        return len(self.img_metadata) if self.split == 'trn' else min(len(self.img_metadata), 1000)
    
    def __getitem__(self, idx):
        idx %= len(self.img_metadata)
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
                'class_name': self.all_class_name[class_sample+1],
                }

        return batch

    def load_frame(self, query_name):
        need_trans = True
        image, mask, image_size = self.data_request(query_name, return_org_size=True, need_trans=need_trans)
        
        aug_times = 0 if self.split=='val' else 1
        images, masks = aug_data([image], [mask], aug_times, self.rand_transform)
        
        return images, masks, image_size
    
    def data_request(self, img_name, return_org_size=False, need_trans=True):
        try:
            mask = self.mask_cache[img_name]
            img = self.img_cache[img_name]
        except KeyError:
            mask = np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png'))[...,None]
            img = np.array(Image.open(os.path.join(self.img_path, img_name) + '.jpg'))
            
            self.mask_cache[img_name] = mask
            self.img_cache[img_name] = img
        
        org_size = img.shape
        if need_trans:
            img, mask = self.crop_resize_trans(img, mask)
        else:
            img, mask = img, mask
        if return_org_size:
            return img, mask, org_size
        else:
            return img, mask

    def sample_episode(self, idx):
        query_name, class_sample = self.img_metadata[idx]
        return query_name, class_sample

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = Path('dataset/splits/pascal') / split / f"fold{fold_id}.txt"
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise
