import numpy as np
import argparse
import time
from apex import amp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.utils
from torchvision import datasets
import torchvision.transforms as transforms
from thop import profile
from timm_models import get_timm_models
from sklearn import metrics
from auto_augment import ImageNetPolicy
from utils import *


def get_args():
	parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
	parser.add_argument('--data', type=str, default='../data/imagenet/', help='path to dataset')
	parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
	parser.add_argument('--model', type=str, required=True, help='model name')
	parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
	parser.add_argument('--learning_rate_min', type=float, default=0., help='min learning rate')
	parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
	parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
	parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
	parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
	parser.add_argument('--classes', type=int, default=1000, help='number of classes')
	parser.add_argument('--epochs', type=int, default=120, help='num of training epochs')
	parser.add_argument('--dropout', type=float, default=0.2, help='drop path probability')
	parser.add_argument('--drop_connect', type=float, default=0.2, help='drop path probability')
	parser.add_argument('--exp_name', type=str, default='train', help='experiment name')
	parser.add_argument('--seed', type=int, default=0, help='random seed')
	parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
	parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
	args = parser.parse_args()
	return args
	
	

def main():
	args = get_args()
	print(args)
	
	np.random.seed(args.seed)
	torch.cuda.set_device(args.gpu)
	cudnn.benchmark = True
	torch.manual_seed(args.seed)
	cudnn.enabled = True
	torch.cuda.manual_seed(args.seed)
	
	# train dataset
	train_transform = transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		ImageNetPolicy(),
		transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		Cutout(16)
    ])
	train_data = datasets.ImageFolder(root='/home/work/dataset/ILSVRC2012/train', transform=train_transform)
	
	# val dataset
	val_transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	valid_data = datasets.ImageFolder(root='/home/work/dataset/ILSVRC2012/train', transform=val_transform)
	
	# data queue
	train_queue = torch.utils.data.DataLoader(
		train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
	
	valid_queue = torch.utils.data.DataLoader(
		valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
	
	# criterion
	criterion = nn.CrossEntropyLoss()
	criterion = criterion.cuda()
	criterion_smooth = CrossEntropyLabelSmooth(args.classes, args.label_smooth)
	criterion_smooth = criterion_smooth.cuda()
	
	# model
	model = get_timm_models(args.model, dropout=args.dropout, drop_connect=args.drop_connect)
	if args.parallel:
		model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
	else:
		model = model.cuda()
		
	# flops, params
	input = torch.randn(1, 3, 224, 224).cuda()
	flops, params = profile(model, inputs=(input,), verbose=False)
	print('{} model, params: {}M, flops: {}M'.format(args.model, params / 1e6, flops / 1e6))
	
	# optimizer
	optimizer = torch.optim.SGD(
		model.parameters(),
		args.learning_rate,
		momentum=args.momentum,
		weight_decay=args.weight_decay
	)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.learning_rate_min)
	
	# Use NVIDIA's apex
	model, optimizer = amp.initialize(model, optimizer)
	
	# train
	best_acc = 0.
	train_top1_list, train_top5_list, train_loss_list = [], [], []
	val_top1_list, val_top5_list, val_loss_list = [], [], []
	for epoch in range(args.epochs):
		time1 = time.time()
		scheduler.step()
		model.drop_path_prob = args.drop_connect * epoch / args.epochs
		
		train_top1, train_top5, train_loss = train(args, epoch, train_queue, model, criterion_smooth, optimizer)
		valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
		
		if valid_acc_top1 > best_acc:
			best_acc = valid_acc_top1
			state = {
				'args': args,
				'best_acc': best_acc,
				'model_state': scheduler.state_dict()
			}
			path = './models/{}_weights.pt.tar'.format(args.exp_name)
			torch.save(state, path)
		
		train_top1_list.append(train_top1)
		train_top5_list.append(train_top5)
		train_loss_list.append(train_loss)
		val_top1_list.append(valid_acc_top1)
		val_top5_list.append(valid_acc_top5)
		val_loss_list.append(valid_obj)
		
		time2 = time.time()
		print(' val loss: {:.6}, val acc: {:.6}, time: {:.3}'.format(valid_obj, valid_acc_top1, time2 - time1))
		print('best val acc: ', best_acc)
	
	# save training process
	state = {
		'train_top1': train_top1_list,
		'train_top5': train_top5_list,
		'train_loss': train_loss_list,
		'val_top1': val_top1_list,
		'val_top5': val_top5_list,
		'val_loss': val_loss_list
	}
	path = './models/{}_train_process.pt.tar'.format(args.exp_name)
	torch.save(state, path)


def train(args, epoch, train_queue, model, criterion, optimizer):
	objs = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	model.train()

	for step, (input, target) in enumerate(train_queue):
		target = target.cuda(async=True)
		input = input.cuda()
		input = Variable(input)
		target = Variable(target)
		
		optimizer.zero_grad()
		logits = model(input)
		loss = criterion(logits, target)
		
		# loss.backward()
		with amp.scale_loss(loss, optimizer) as scaled_loss:
			scaled_loss.backward()
		nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
		optimizer.step()
		
		prec1, prec5 = accuracy(logits, target, topk=(1, 5))
		n = input.size(0)
		objs.update(loss.data[0], n)
		top1.update(prec1.data[0], n)
		top5.update(prec5.data[0], n)
		
		print('\rEpoch: {}/{}, step[{}/{}], train loss: {:.6}, train top1: {:.6}, train top5: {:.6}'.format(
			epoch + 1, args.epochs, step + 1, len(train_queue), objs.avg, top1.avg, top5.avg), end='')
	return top1.avg, top5.avg, objs.avg
	

def infer(valid_queue, model, criterion):
	objs = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	model.eval()
	
	for step, (input, target) in enumerate(valid_queue):
		input = Variable(input, volatile=True).cuda()
		target = Variable(target, volatile=True).cuda(async=True)
		
		logits = model(input)
		loss = criterion(logits, target)
		
		prec1, prec5 = accuracy(logits, target, topk=(1, 5))
		n = input.size(0)
		objs.update(loss.data[0], n)
		top1.update(prec1.data[0], n)
		top5.update(prec5.data[0], n)
	
	return top1.avg, top5.avg, objs.avg
	
	
if __name__ == '__main__':
	main()
	