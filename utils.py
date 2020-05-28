import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from auto_augment import ImageNetPolicy


class AverageMeter(object):
	def __init__(self):
		self.reset()
	
	def reset(self):
		self.avg = 0
		self.sum = 0
		self.cnt = 0
	
	def update(self, val, n=1):
		self.sum += val * n
		self.cnt += n
		self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)
	
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


class CrossEntropyLabelSmooth(nn.Module):
	def __init__(self, num_classes, epsilon):
		super(CrossEntropyLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.logsoftmax = nn.LogSoftmax(dim=1)
	
	def forward(self, inputs, targets):
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (-targets * log_probs).mean(0).sum()
		return loss


class Cutout(object):
	def __init__(self, length):
		self.length = length
	
	def __call__(self, img):
		h, w = img.size(1), img.size(2)
		mask = np.ones((h, w), np.float32)
		y = np.random.randint(h)
		x = np.random.randint(w)
		
		y1 = np.clip(y - self.length // 2, 0, h)
		y2 = np.clip(y + self.length // 2, 0, h)
		x1 = np.clip(x - self.length // 2, 0, w)
		x2 = np.clip(x + self.length // 2, 0, w)
		
		mask[y1: y2, x1: x2] = 0.
		mask = torch.from_numpy(mask)
		mask = mask.expand_as(img)
		img *= mask
		return img


def Imagenet_transforms():
	train_transform = transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		ImageNetPolicy(),
		transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		Cutout(16)
	])
	
	val_transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	
	return train_transform, val_transform


class data_prefetcher():
	def __init__(self, loader):
		self.loader = iter(loader)
		self.stream = torch.cuda.Stream()
		self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
		self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
		# With Amp, it isn't necessary to manually convert data to half.
		# if args.fp16:
		#     self.mean = self.mean.half()
		#     self.std = self.std.half()
		self.preload()

	def preload(self):
		try:
			self.next_input, self.next_target = next(self.loader)
		except StopIteration:
			self.next_input = None
			self.next_target = None
			return
		# if record_stream() doesn't work, another option is to make sure device inputs are created
		# on the main stream.
		# self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
		# self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
		# Need to make sure the memory allocated for next_* is not still in use by the main stream
		# at the time we start copying to next_*:
		# self.stream.wait_stream(torch.cuda.current_stream())
		with torch.cuda.stream(self.stream):
			self.next_input = self.next_input.cuda(non_blocking=True)
			self.next_target = self.next_target.cuda(non_blocking=True)
			# more code for the alternative if record_stream() doesn't work:
			# copy_ will record the use of the pinned source tensor in this side stream.
			# self.next_input_gpu.copy_(self.next_input, non_blocking=True)
			# self.next_target_gpu.copy_(self.next_target, non_blocking=True)
			# self.next_input = self.next_input_gpu
			# self.next_target = self.next_target_gpu
			
			# With Amp, it isn't necessary to manually convert data to half.
			# if args.fp16:
			#     self.next_input = self.next_input.half()
			# else:
			self.next_input = self.next_input.float()
			self.next_input = self.next_input.sub_(self.mean).div_(self.std)

	def next(self):
		torch.cuda.current_stream().wait_stream(self.stream)
		input = self.next_input
		target = self.next_target
		if input is not None:
			input.record_stream(torch.cuda.current_stream())
		if target is not None:
			target.record_stream(torch.cuda.current_stream())
		self.preload()
		return input, target

