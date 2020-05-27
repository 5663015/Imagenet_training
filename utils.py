import numpy as np
import torch
import torch.nn as nn


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


