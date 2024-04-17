import torch
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN

from data.datasets import build_dataset
from data.samplers import RASampler
from utils import utils

def getDataLoader(args, dataset_type=0):
    if dataset_type == 0:
        dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
        dataset_val, _ = build_dataset(is_train=False, args=args)

        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)

        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=True
        )

        loader_eval = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(3 * args.batch_size),
            shuffle=False, num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=True
        )
    elif dataset_type == 1:
        from timm.data import create_dataset, create_loader, AugMixDataset
        dataset_train = create_dataset(args.dataset, root=args.data_path, split="train", is_training=True, download=False)
        if args.aug_splits > 1: dataset_train = AugMixDataset(dataset_train, num_splits=args.aug_splits)

        loader_train = create_loader(
            dataset_train,
            input_size=args.input_size,
            batch_size=args.batch_size,
            is_training=True,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            num_aug_splits=args.aug_splits,
            interpolation=args.train_interpolation,
            mean=IMAGENET_INCEPTION_MEAN,
            std=IMAGENET_INCEPTION_STD,
            num_workers=args.num_workers,
            distributed=args.distributed,
            pin_memory=args.pin_mem
        )

        dataset_eval = create_dataset(args.dataset, root=args.data_path, split="val", is_training=False, download=False)

        loader_eval = create_loader(
            dataset_eval,
            input_size = args.input_size,
            batch_size = 4*args.batch_size,
            is_training=False,
            interpolation='bicubic',
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            num_workers=args.num_workers,
            distributed=args.distributed,
            crop_pct=0.9,
            pin_memory=args.pin_mem,
        )
    else:
        raise ValueError

    return loader_train, loader_eval