import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', default='coco', type=str)
parser.add_argument('--dataset_root', default='./data', type=str)
parser.add_argument('--fold', default=0, type=int)

parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--aug_prob', default=0.5, type=float)
parser.add_argument('--test_with_org_resolution', default=False, action='store_true')
parser.add_argument('--target_size', default=[224, 224], type=int, nargs='+')
parser.add_argument('--scale_range', default=[0.5, 1.0], type=float, nargs='+')

parser.add_argument('--warmup_epochs', default=10, type=int)
parser.add_argument('--training_epochs', default=500, type=int)
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--fusion_lr', default=0.001, type=float)
parser.add_argument('--fusion_min_lr', default=2.0e-6, type=float)
parser.add_argument('--decoder_lr', default=0.001, type=float)
parser.add_argument('--wd', default=0.01, type=float)
parser.add_argument('--wd_end', default=0.4, type=float)

parser.add_argument('--drop_path_rate', default=0., type=float)
parser.add_argument('--transformer_dropout', default=0., type=float)
parser.add_argument('--transformer_heads', default=8, type=int)
parser.add_argument('--transformer_layers', default=12, type=int)
parser.add_argument('--transformer_width', default=512, type=int)
parser.add_argument('--temperature', default=0.07, type=float)
parser.add_argument('--load_ckpt_path', default=None, type=str)
    
parser.add_argument('--device', default='cuda', type=str)

args = parser.parse_args()

