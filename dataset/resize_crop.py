import albumentations as A
import cv2

from configs import args


class Resize:
    def __init__(self, target_size) -> None:
        self.target_size = (target_size, target_size) if isinstance(target_size, (int, float)) else target_size 
        self.img_rsz = A.Resize(*self.target_size, interpolation=cv2.INTER_AREA)
    
    def __call__(self, img, gt):
        ret_dict = self.img_rsz(image=img, mask=gt)
        img = ret_dict['image']
        gt = ret_dict['mask']
        return img, gt

class RandomCropResize:
    def __init__(self, target_size, scale_range) -> None:
        if isinstance(target_size, (int, float)):
            target_size = [target_size, target_size]
        if scale_range is None:
            scale_range = [0.5, 1.0]
        self.resize_crop = A.RandomResizedCrop(target_size[1], target_size[0], scale=scale_range, p=args.aug_prob)

        self.only_crop = A.Sequential(
            [A.SmallestMaxSize(target_size[0], p=1), 
            A.RandomCrop(target_size[1], target_size[0], p=1)], p=1-args.aug_prob)

        self.trans = A.OneOf([self.resize_crop, self.only_crop], p=1)

    def __call__(self, img, gt):
        ret_dict = self.trans(image=img, mask=gt)
        return ret_dict['image'], ret_dict['mask']
