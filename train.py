'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import sys
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from timm_models import get_timm_models
from utils.dataloaders import *
from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/home/work/dataset/ILSVRC2012', help='path to dataset')
parser.add_argument('--data_backend', metavar='BACKEND', default='pytorch', choices=DATA_BACKEND_CHOICES)
parser.add_argument('--arch', metavar='ARCH', required=True, help='model architecture')
parser.add_argument('--workers', default=32, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=256, type=int,metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--bn_momentum', type=float, default=0.9, help='BatchNorm momentum override (if not None)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,metavar='W',
                    help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--dropout', type=float, default=0.2, help='drop path probability')
parser.add_argument('--drop_connect', type=float, default=0.2, help='drop path probability')
parser.add_argument('--print_freq', default=1, type=int, metavar='N', help='print frequency (default: 1)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--lr_decay', type=str, default='cos', help='mode for learning rate decay')
parser.add_argument('--step', type=int, default=30, help='interval for learning rate decay in step mode')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--warmup', action='store_true', default=False, help='set lower initial learning rate to warm up the training')
parser.add_argument('--checkpoint', default='./checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--width_mult', type=float, default=1.0, help='MobileNet model width multiplier.')
parser.add_argument('--input_size', type=int, default=224, help='MobileNet model input resolution')

best_prec1 = 0


def main():
    # prepare dir
    if not os.path.exists('./logdir'):
        os.mkdir('./logdir')

    global args, best_prec1
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    assert args.arch in ['efficientnet_b0', 'mixnet_m', 'mobilenet_v3', 'mnasnet_a1']
    print("=> creating model '{}'".format(args.arch))
    model = get_timm_models(args.arch, dropout=args.dropout, drop_connect=args.drop_connect, bn_momentum=args.bn_momentum)

    if not args.distributed:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    title = 'ImageNet-' + args.arch
    logger = Logger(os.path.join(args.checkpoint, '{}_log.txt'.format(args.arch)), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    cudnn.benchmark = True

    # Data loading code
    if args.data_backend == 'pytorch':
        get_train_loader = get_pytorch_train_loader
        get_val_loader = get_pytorch_val_loader
    elif args.data_backend == 'dali_gpu':
        get_train_loader = get_dali_train_loader(dali_cpu=False)
        get_val_loader = get_dali_val_loader()
    elif args.data_backend == 'dali_cpu':
        get_train_loader = get_dali_train_loader(dali_cpu=True)
        get_val_loader = get_dali_val_loader()

    train_loader, train_loader_len = get_train_loader(args.data, args.batch_size, workers=args.workers,
                                                      input_size=args.input_size)
    val_loader, val_loader_len = get_val_loader(args.data, args.batch_size, workers=args.workers,
                                                input_size=args.input_size)

    # visualization
    writer = SummaryWriter(os.path.join(args.checkpoint, '{}_logs'.format(args.arch)))

    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()

        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_loss, train_acc = train(train_loader, train_loader_len, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_loss, prec1 = validate(val_loader, val_loader_len, model, criterion)
        elapse = time.time() - t1
        h, m, s = eta_time(elapse, args.epochs - epoch - 1)

        lr = optimizer.param_groups[0]['lr']

        # append logger file
        logger.append([lr, train_loss, val_loss, train_acc, prec1])

        # tensorboardX
        writer.add_scalar('learning rate', lr, epoch + 1)
        writer.add_scalars('loss', {'train loss': train_loss, 'validation loss': val_loss}, epoch + 1)
        writer.add_scalars('accuracy', {'train accuracy': train_acc, 'validation accuracy': prec1}, epoch + 1)

        if prec1 > best_prec1:
            best_prec1 = max(prec1, best_prec1)
            torch.save(obj={
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, f=os.path.join(args.checkpoint, '{}.pth.tar'.format(args.arch)))
        print('\nEpoch: [{:.0f} | {:.0f}], val: loss={:.6}, top1={:.6}, best=\033[31m{:.6}\033[0m, elapse={:.0f}s, eta={:.0f}h {:.0f}m {:.0f}s'
              .format(epoch + 1, args.epochs, val_loss, prec1, best_prec1, elapse, h, m, s))

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, '{}_log.eps'.format(args.arch)))
    writer.close()

    print('Best accuracy:')
    print(best_prec1)


def train(train_loader, train_loader_len, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, i, train_loader_len)

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        sys.stdout.write(
            '\r({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=i + 1,
                size=train_loader_len,
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg)
        )
        sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, val_loader_len, model, criterion):
    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # measure data loading time
        # data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        # # plot progress
        # sys.stdout.write(
        #     '\r({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
        #         batch=i + 1,
        #         size=val_loader_len,
        #         data=data_time.avg,
        #         bt=batch_time.avg,
        #         loss=losses.avg,
        #         top1=top1.avg,
        #         top5=top5.avg)
        # )
        # sys.stdout.flush()

    return losses.avg, top1.avg


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def eta_time(elapse, epoch):
    eta = epoch * elapse
    hour = eta // 3600
    minute = (eta - hour * 3600) // 60
    second = eta - hour * 3600 - minute * 60
    return hour, minute, second


from math import cos, pi


def adjust_learning_rate(optimizer, epoch, iteration, num_iter):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5 if args.warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** ((current_iter - warmup_iter) // (max_iter - warmup_iter)))
    elif args.lr_decay == 'cos':
        lr = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args.lr_decay == 'schedule':
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = args.lr * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    if epoch < warmup_epoch:
        lr = args.lr * current_iter / warmup_iter

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
