import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from pathlib import Path
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from datasets import build_dataset
from losses import DistillationLoss
from samplers import RASampler
import models
import utils
import sys
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from engine import evaluate
from timm.utils import accuracy
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize
from compute_flops import compute_flops
import copy


num_blocks = 12


# nn.Linear indices
attn_qkv = [4*i+1 for i in range(num_blocks)]
attn_proj = [4*i+2 for i in range(num_blocks)]
mlp_fc1 = [4*i+3 for i in range(num_blocks)]
mlp_fc2 = [4*i+4 for i in range(num_blocks)]


def mlp_neuron_rank(model, train_loader):
	relevance = HSICLoss(y_kernel='linear', mean_sub=True).cuda()
	redundancy = HSICLoss(y_kernel='rbf', mean_sub=False).cuda()
	score = {}
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(train_loader):
			if batch_idx >= 1:
				break
			data, target = Variable(data), Variable(target)
			data, target = data.cuda(), target.cuda()
			output = model(data)
			idx = 0
			for m in model.modules():
				if 'Mlp' in str(m) and 'Attention' not in str(m):
					X_ = m.neuron_output # batch x seq x embed
					hsic = []
					for H1 in range(X_.shape[-1]):
						hsic.append(relevance(X_[:,:,H1], F.softmax(output, dim=-1)).item())
					hsic = np.array(hsic)
					hsic = (hsic - np.min(hsic)) / (np.max(hsic) - np.min(hsic))
					act = np.sum(X_.abs().detach().cpu().numpy(), axis=(0,1))
					act = (act - np.min(act)) / (np.max(act) - np.min(act))
					temp = (0.1*hsic + 0.9*act).tolist()
					if batch_idx == 0:
						score[str(idx)] = np.array(temp)
					else:
						score[str(idx)] += np.array(temp)
					idx += 1
					continue
				
	# rank是一个列表，其中每个元素是模型中每一层MLP排好序的神经元索引列表。
	rank = [np.argsort(score[str(idx)]) for idx in range(len(score))]

	# data=open("../Data_FFN0.txt",'a') 
	# print("FFN_score:\n")
	# for idx in range(len(score)):
	# 	data.write(str(score[str(idx)])+'\n')
	# 	print(score[str(idx)],end='\n')
	# print("FFN_rank:\n")
	# for idx in range(len(rank)):
	# 	data.write(str(rank[idx])+'\n')
	# 	print(rank[idx],end='\n')
	# data.close()

	return rank


def mlp_neuron_mask(model, ratio, rank):
	idx = 0
	neuron_mask = []
	for m in model.modules():
		if 'Mlp' in str(m) and 'Attention' not in str(m):
			num_keep = int(m.hidden_features * (1 - ratio[idx]))
			# [::-1]是返回逆序列表的意思
			arg_max_rev = rank[idx][::-1][:num_keep]
			mask = torch.zeros(m.hidden_features)
			mask[arg_max_rev.tolist()] = 1
			neuron_mask.append(mask)
			idx += 1
			continue
	return neuron_mask


def mlp_neuron_prune(model, neuron_mask):
	idx = 0
	for m in model.modules():
		if 'Mlp' in str(m) and 'Attention' not in str(m):
			m.gate = neuron_mask[idx]
			idx += 1
			continue

# 撤销剪枝操作
def mlp_neuron_restore(model, ):
	idx = 0
	for m in model.modules():
		if 'Mlp' in str(m) and 'Attention' not in str(m):
			temp = m.gate.detach().clone()
			m.gate = torch.ones(temp.shape[0])
			idx += 1
			continue

# 查看剪枝率
def check_neuron_sparsity(model):
	ratio = []
	for m in model.modules():
		if 'Mlp' in str(m) and 'Attention' not in str(m):
			ratio.append(torch.sum(m.gate == 0).item() / m.gate.shape[0])
			continue
	return ratio


def attn_head_rank(model, train_loader):
	relevance = HSICLoss(y_kernel='linear', mean_sub=True).cuda()
	redundancy = HSICLoss(y_kernel='rbf', mean_sub=False).cuda()
	score = {}
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(train_loader):
			if batch_idx >= 1:
				break
			data, target = Variable(data), Variable(target)
			data, target = data.cuda(), target.cuda()
			output = model(data)
			idx = 0
			for m in model.modules():
				if 'Attention' in str(m) and 'Mlp' not in str(m):
					X_ = m.head_output # batch x seq x head x embed_chunk
					temp = []
					for H1 in range(X_.shape[2]):
						# max relevance
						relevance_count = relevance(torch.mean(X_[:,:,H1,:], dim=-1), F.softmax(output, dim=-1)).item()
						# min redundancy
						redundancy_count = 0
						for H2 in range(X_.shape[2]):
							if H2 != H1:
								redundancy_count += redundancy(torch.mean(X_[:,:,H1,:], dim=-1), torch.mean(X_[:,:,H2,:], dim=-1)).item()
						redundancy_count /= (X_.shape[2]-1)
						temp.append(relevance_count - 0.1*redundancy_count)
					if batch_idx == 0:
						score[str(idx)] = np.array(temp)
					else:
						score[str(idx)] += np.array(temp)
					idx += 1
					continue
	rank = [(np.argsort(score[str(idx)])) for idx in range(len(score))]

	# data=open("../Data_MHSA0.txt",'a') 
	# print("MHSA_score:\n")
	# for idx in range(len(score)):
	# 	data.write(str(score[str(idx)])+'\n')
	# 	print(score[str(idx)],end='\n')
	# print("MHSA_rank:\n")
	# for idx in range(len(rank)):
	# 	data.write(str(rank[idx])+'\n')
	# 	print(rank[idx],end='\n')
	# data.close()

	return rank


def attn_head_mask(model, ratio, rank):
	idx = 0
	head_mask = []
	for m in model.modules():
		if 'Attention' in str(m) and 'Mlp' not in str(m):
			num_keep = int(m.num_heads * (1 - ratio[idx]))
			arg_max_rev = rank[idx][::-1][:num_keep]
			mask = torch.zeros(m.num_heads)
			mask[arg_max_rev.tolist()] = 1
			head_mask.append(mask)
			idx += 1
			continue
	return head_mask


def attn_head_prune(model, head_mask):
	idx = 0
	for m in model.modules():
		if 'Attention' in str(m) and 'Mlp' not in str(m):
			m.gate = head_mask[idx]
			idx += 1
			continue


def attn_head_restore(model, ):
	idx = 0
	for m in model.modules():
		if 'Attention' in str(m) and 'Mlp' not in str(m):
			temp = m.gate.detach().clone()
			m.gate = torch.ones(temp.shape[0])
			idx += 1
			continue


def check_head_sparsity(model):
	ratio = []
	for m in model.modules():
		if 'Attention' in str(m) and 'Mlp' not in str(m):
			ratio.append(torch.sum(m.gate == 0).item() / m.gate.shape[0])
			continue
	return ratio


def token_layer_rank(model, train_loader):
	model_select = copy.deepcopy(model)
	output = None
	score = np.inf
	layer = []
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(train_loader):
			if batch_idx >= 1:
				break
			data, target = Variable(data), Variable(target)
			data, target = data.cuda(), target.cuda()
			output = model(data)
	for idx in np.ndindex(3, 3, 3):
		tmp = np.add(list(idx), [4, 7, 10])
		tmp = list(tmp)
		set_token_selection_layer(model_select, 0.3, tmp)
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(train_loader):
				if batch_idx >= 1:
					break
				data, target = Variable(data), Variable(target)
				data, target = data.cuda(), target.cuda()
				output_select = model_select(data)
				score_select = kl_divergence(F.softmax(output, dim=-1), F.softmax(output_select, dim=-1)).item()
				if score_select < score:					
					score = score_select
					layer = tmp
					# print("layer:", layer)
		reset_token_selection_layer(model_select, tmp)

	return layer
				

def kl_divergence(p, q):
    """
    计算两个概率分布p和q之间的KL散度
    :param p: 第一个概率分布 (Tensor)
    :param q: 第二个概率分布 (Tensor)
    :return: KL散度 (Tensor)
    """
    p = p / p.sum()
    q = q / q.sum()
    return F.kl_div(p.log(), q, reduction='batchmean')


def set_token_selection_layer(model, token_sparsity, layer=[4,7,10]):
	idx = 1
	for m in model.modules():
		if 'Attention' in str(m) and 'Mlp' not in str(m):
			if idx in layer:
				m.token_prune_ratio = token_sparsity
			idx += 1
			continue

def reset_token_selection_layer(model, layer=[4,7,10]):
	idx = 1
	for m in model.modules():
		if 'Attention' in str(m) and 'Mlp' not in str(m):
			if idx in layer:
				m.token_prune_ratio = 0
			idx += 1
			continue	
	for m in model.modules():
		if 'Block' in str(m) and 'ModuleList' not in str(m):
			m.ema_cls_attn = None


def center(X):
    mean_col = torch.mean(X, dim=0, keepdim=True)
    mean_row = torch.mean(X, dim=1, keepdim=True)
    mean_all = torch.mean(X)
    return X - mean_col - mean_row + mean_all
    

class GaussianKernel(nn.Module):
    def __init__(self, sigma):
        super(GaussianKernel, self).__init__()
        assert sigma > 0
        self.sigma = sigma

    def forward(self, x):
        X_inner = torch.matmul(x, x.t())
        X_norm = torch.diag(X_inner, diagonal=0)
        X_dist_sq = X_norm + torch.reshape(X_norm, [-1,1]) - 2 * X_inner
        return torch.exp( - X_dist_sq / (2 * self.sigma**2))


class LinearKernel(nn.Module):
    def __init__(self,):
        super(LinearKernel, self).__init__()

    def forward(self, x):
        return torch.matmul(x, x.t())
    

class HSICLoss(nn.Module):
    def __init__(self, y_kernel='linear', mean_sub=False):
        super(HSICLoss, self).__init__()

        self.kernelX_1 = GaussianKernel(1)
        self.kernelX_2 = GaussianKernel(2)
        self.kernelX_4 = GaussianKernel(4)
        self.kernelX_8 = GaussianKernel(8)
        self.kernelX_16 = GaussianKernel(16)

        self.y_kernel = y_kernel
        if self.y_kernel == 'linear':
            self.kernelY = LinearKernel()
        elif self.y_kernel == 'rbf':
            self.kernelY = None

        self.mean_sub = mean_sub

    def forward(self, x, y):
        '''
        x: feature
        y: softmax prediction
        '''
        if self.mean_sub is True:
            x = x - torch.mean(x, dim=0) / (torch.std(x, dim=0) + 1e-12)
            y = y - torch.mean(y, dim=0)

        G_X = center((self.kernelX_1(x) + self.kernelX_2(x) + self.kernelX_4(x) + self.kernelX_8(x) + self.kernelX_16(x))/5)

        if self.y_kernel == 'linear':
            G_Y = center(self.kernelY(y))
        elif self.y_kernel == 'rbf':
            G_Y = center((self.kernelX_1(y) + self.kernelX_2(y) + self.kernelX_4(y) + self.kernelX_8(y) + self.kernelX_16(y))/5)

        return torch.trace(torch.matmul(G_X, G_Y))