"""Training arguments, learning rate schedule and checkpoint helpers."""
import argparse
from pathlib import Path
import shutil

import torch


def parse_command(argv=None):
    parser = argparse.ArgumentParser(description='Individual opinion prediction training')
    parser.add_argument('--data-root', required=True, help='Dataset directory containing stimuli/')
    parser.add_argument('--annotations', required=True, help='Path to the local CUID observer-score workbook')
    parser.add_argument('--image-subdir', default='stimuli')
    parser.add_argument('--image-suffix', default='.bmp', help='Use an empty string to retain workbook extensions')
    parser.add_argument('--train-ratio', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arch', '-a', choices=['MANIQA', 'resnet18', 'resnet50', 'vgg16'], default='MANIQA')
    parser.add_argument('--decoder', '-d', choices=['deconv2', 'deconv3', 'upconv', 'upproj'], default='deconv2')
    parser.add_argument('--criterion', '-c', choices=['l1', 'l2'], default='l2')
    parser.add_argument('--workers', '-j', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', '-b', type=int, default=4)
    parser.add_argument('--lr', '--learning-rate', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', '--wd', type=float, default=1e-4)
    parser.add_argument('--print-freq', '-p', type=int, default=10)
    parser.add_argument('--resume', default='')
    parser.add_argument('--evaluate', '-e', default='', help='Evaluate a state-dict checkpoint')
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--no-pretrain', dest='pretrained', action='store_false')
    parser.set_defaults(pretrained=True)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.epochs < 1 or args.workers < 0 or args.print_freq < 1 or args.lr <= 0:
        parser.error('Batch size, epochs, print frequency and learning rate must be positive; workers must be nonnegative')
    if args.resume and args.evaluate:
        parser.error('Use either --resume or --evaluate')
    return args


def save_checkpoint(state, is_best, epoch, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'checkpoint.pth.tar'
    temporary = output / 'checkpoint.tmp'
    torch.save(state, temporary)
    temporary.replace(path)
    if is_best:
        shutil.copyfile(path, output / 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, lr_init):
    """Reduce the learning rate by a factor of 10 every 50 epochs."""
    for group in optimizer.param_groups:
        group['lr'] = lr_init * (0.1 ** (epoch // 50))


def get_output_directory(args):
    return args.output_dir
