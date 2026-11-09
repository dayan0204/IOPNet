import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import criteria
from datasets.dataloader import TID2013DatasetLoader
from metrics import AverageMeter, Result
from models import MANIQA, ResNet, VGGModel
import utils


fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae', 'delta1', 'delta2',
              'delta3', 'distribution_error', 'data_time', 'gpu_time']


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def create_data_loaders(args, device, generator):
    common = dict(root=args.data_root, gt_file=args.annotations, img_size=(224, 224),
                  seed=args.seed, global_permute=True, train_ratio=args.train_ratio,
                  image_subdir=args.image_subdir, image_suffix=args.image_suffix)
    validation = TID2013DatasetLoader(**common, train_phase=False, is_sample=False)
    val_loader = DataLoader(validation, batch_size=1, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == 'cuda',
                            worker_init_fn=seed_worker)
    train_loader = None
    if not args.evaluate:
        training = TID2013DatasetLoader(**common, train_phase=True, is_sample=True)
        if len(training) < args.batch_size:
            raise ValueError('Training split must contain at least batch-size images (drop_last=True)')
        train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, drop_last=True,
                                  pin_memory=device.type == 'cuda', worker_init_fn=seed_worker,
                                  generator=generator)
    return train_loader, val_loader


def create_model(args, pretrained=None):
    use_pretrained = args.pretrained if pretrained is None else pretrained
    if args.arch == 'MANIQA':
        return MANIQA(embed_dim=72, num_outputs=19, img_size=224, pretrained=use_pretrained)
    if args.arch in ('resnet18', 'resnet50'):
        return ResNet(layers=int(args.arch[6:]), decoder=args.decoder, output_size=(1, 19),
                      in_channels=3, pretrained=use_pretrained)
    return VGGModel(load_weight=int(use_pretrained))


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def append_result(path, result):
    path = Path(path)
    new = not path.exists()
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if new:
            writer.writeheader()
        writer.writerow({field: getattr(result, field) for field in fieldnames})


def train(train_loader, model, criterion, distribution_criterion, optimizer, epoch, args, device):
    meter = AverageMeter()
    model.train()
    end = time.perf_counter()
    for index, (images, _, target, _) in enumerate(train_loader):
        images, target = images.to(device), target.to(device)
        synchronize(device)
        data_time = time.perf_counter() - end
        start = time.perf_counter()
        pred = model(images)
        loss = criterion(pred, target) + distribution_criterion(pred, target)
        if not torch.isfinite(loss):
            raise RuntimeError('Training loss is nonfinite')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        synchronize(device)
        gpu_time = time.perf_counter() - start
        result = Result()
        result.evaluate(pred.detach(), target.detach())
        meter.update(result, gpu_time, data_time, images.shape[0])
        if (index + 1) % args.print_freq == 0:
            print(f'Train epoch {epoch} [{index+1}/{len(train_loader)}] '
                  f'loss={loss.item():.6f} rmse={meter.average().rmse:.6f}', flush=True)
        end = time.perf_counter()
    average = meter.average()
    append_result(Path(args.output_dir) / 'train.csv', average)
    return average


@torch.no_grad()
def validate(val_loader, model, args, device, write_to_file=True):
    meter = AverageMeter()
    model.eval()
    end = time.perf_counter()
    for images, _, target, _ in val_loader:
        images, target = images.to(device), target.to(device)
        synchronize(device)
        data_time = time.perf_counter() - end
        start = time.perf_counter()
        pred = model(images)
        synchronize(device)
        gpu_time = time.perf_counter() - start
        result = Result()
        result.evaluate(pred, target)
        meter.update(result, gpu_time, data_time, images.shape[0])
        end = time.perf_counter()
    average = meter.average()
    print(f'Validation mse={average.mse:.6f} rmse={average.rmse:.6f} '
          f'delta1={average.delta1:.6f} distribution_error={average.distribution_error:.6f}', flush=True)
    if write_to_file:
        append_result(Path(args.output_dir) / 'test.csv', average)
    return average


def main(argv=None):
    args = utils.parse_command(argv)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    checkpoint = None
    if args.resume or args.evaluate:
        checkpoint = torch.load(args.resume or args.evaluate, map_location='cpu', weights_only=True)
        if checkpoint.get('format_version') != 1:
            raise ValueError('Use a release state-dict checkpoint; legacy full-object checkpoints need explicit migration')
        for key, value in checkpoint['args'].items():
            if key not in {'data_root', 'annotations', 'output_dir', 'device', 'resume', 'evaluate',
                           'epochs', 'workers', 'print_freq'}:
                setattr(args, key, value)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator().manual_seed(args.seed)
    model = create_model(args, pretrained=False if checkpoint else None).to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    best_result = Result()
    best_result.set_to_worst()
    start_epoch = 0
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        if not args.evaluate:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_result.__dict__.update(checkpoint['best_result'])
        random.setstate(checkpoint['python_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        generator.set_state(checkpoint['loader_rng'])
        if device.type == 'cuda' and checkpoint['cuda_rng']:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    train_loader, val_loader = create_data_loaders(args, device, generator)
    if args.evaluate:
        return validate(val_loader, model, args, device, write_to_file=False)
    if start_epoch >= args.epochs:
        raise ValueError('Checkpoint has already reached the requested total epochs')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((output / name).exists() for name in ('train.csv', 'test.csv', 'checkpoint.pth.tar')):
        raise ValueError('Output directory already contains a run; choose a new directory or use --resume')
    criterion = criteria.MaskedMSELoss() if args.criterion == 'l2' else criteria.MaskedL1Loss()
    distribution_criterion = criteria.DistributionLoss()
    print(f'Model={args.arch} device={device} train={len(train_loader.dataset)} val={len(val_loader.dataset)}')
    for epoch in range(start_epoch, args.epochs):
        utils.adjust_learning_rate(optimizer, epoch, args.lr)
        train(train_loader, model, criterion, distribution_criterion, optimizer, epoch, args, device)
        result = validate(val_loader, model, args, device)
        is_best = result.rmse < best_result.rmse
        if is_best:
            best_result = result
        utils.save_checkpoint({'format_version': 1, 'args': vars(args), 'epoch': epoch,
                               'model_state_dict': model.state_dict(),
                               'optimizer_state_dict': optimizer.state_dict(),
                               'best_result': vars(best_result), 'python_rng': random.getstate(),
                               'torch_rng': torch.get_rng_state(), 'loader_rng': generator.get_state(),
                               'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
                              is_best, epoch, args.output_dir)


if __name__ == '__main__':
    main()
