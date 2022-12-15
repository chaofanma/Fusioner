from torch.utils.data import DataLoader

from configs import args

from .augumenter import Augumenter
from .custom_dataset import CustomDataset, collect_fn


def get_dataset():
    aug_train = Augumenter()
    aug_test = Augumenter()
    ds_train = CustomDataset('train', aug_train)
    ds_test = CustomDataset('val', aug_test)
    
    dl_train = DataLoader(ds_train, args.batch_size, shuffle=True, collate_fn=collect_fn)
    if args.test_with_org_resolution:
        dl_test = DataLoader(ds_test, 1, shuffle=True, collate_fn=collect_fn)
    else:
        dl_test = DataLoader(ds_test, args.batch_size, shuffle=True, collate_fn=collect_fn)
    return dl_train, dl_test

