import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from configs import args


class Augumenter:
    def __init__(self) -> None:
        
        self._border_args = {
            'border_mode': cv2.BORDER_CONSTANT,
            'value': None,
            'mask_value': None, 
        }

        aug_list = []
        
        if args.dataset_name == 'pascal':
            gt_invarient_list = {
                'color-jitter': A.ColorJitter(0.5, 0.5, 0.5, 0.5, p=0.5),
                'gauss-noise':A.GaussNoise(p=0.5),
                'h-flip': A.HorizontalFlip(p=0.5),
                'elastic': A.ElasticTransform(alpha_affine=20, p=0.5, **self._border_args),
                'affine': A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5, **self._border_args),
            }
        else:
            gt_invarient_list = {
                'color-jitter': A.ColorJitter(0.2, 0.2, 0.2, 0.2, p=0.5),
                'gauss-noise':A.GaussNoise(p=0.5),
                'h-flip': A.HorizontalFlip(p=0.5),
                'elastic': A.ElasticTransform(alpha_affine=20, p=0.5, **self._border_args),
                'affine': A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5, **self._border_args),
            }
        for key in ('color-jitter', 'gauss-noise', 'h-flip', 'elastic', 'affine'):
            try:
                aug_list.append(gt_invarient_list[key])
            except KeyError:
                print(f"{key} is not a supported augmentation, it is ignored")
        
        self.augs = A.Compose(aug_list, p=args.aug_prob)
        
        if args.dataset_name == 'pascal':
            norm = A.Normalize(mean=[0.4554, 0.4298, 0.3955], std=[0.1181, 0.1189, 0.1338])
        elif args.dataset_name == 'coco':
            norm = A.Normalize(mean=[0.4650, 0.4357, 0.3973], std=[0.1152, 0.1153, 0.1352])
        self.format_trans = A.Compose([
            norm, 
            A.ToFloat(),
            ToTensorV2(transpose_mask=True),
        ], p=1)
    
    def __call__(self, image, mask, do_aug=True):
        img_trans, gt_trans = image, mask
        if do_aug:
            ret_dict = self.augs(image=img_trans, mask=gt_trans)
        else:
            ret_dict = {'image':img_trans, 'mask':gt_trans}
        ret_dict = self.format_trans(**ret_dict)
        return ret_dict
