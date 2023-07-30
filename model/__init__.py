import torch
from torch import optim

from configs import args
from utils import utils

from .full_model import FullModel


def get_learner(dataloader_len):
    
    target_size = args.target_size if isinstance(args.target_size, (int, float)) else args.target_size[0]
    eval_args = {
        'pascal':{'base_size': 500, 'crop_size': target_size, 'flip':False, 'scales':[1.0]}, 
        'coco' :{'base_size': 500, 'crop_size': target_size, 'flip':False, 'scales':[1.0]},
    }

    model = FullModel(**eval_args[args.dataset_name]).to(args.device)

    model.img_model.requires_grad_(False)
    model.text_model.requires_grad_(False)
    
    fusion_params_groups = utils.get_params_groups(model.fusion)
    decoder_params_groups = utils.get_params_groups(model.decoder)
    
    fusion_lr_schedule = utils.cosine_scheduler(
        base_value=args.fusion_lr,
        final_value=args.fusion_min_lr,
        epochs=args.epochs, 
        niter_per_ep=dataloader_len,
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        base_value=args.wd,
        final_value=args.wd_end,
        epochs=args.epochs, 
        niter_per_ep=dataloader_len,
    )
    
    schedule = {
        'fusion': fusion_lr_schedule,
        'wd': wd_schedule,
    }
    
    optimizers = utils.Optimizers({
        'fusion': optim.AdamW(fusion_params_groups, betas=(args.beta1, 0.999)) if fusion_params_groups else None,
        'decoder': optim.Adam(decoder_params_groups, lr=args.decoder_lr) if decoder_params_groups else None,
    }, schedules_dict=schedule)
    
    if args.load_ckpt_path:
        print(f"Loading model from {args.load_ckpt_path}")
        state_dict = torch.load(args.load_ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict['model'], strict=True)
            
    return model, optimizers
