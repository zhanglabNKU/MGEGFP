import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

import args






def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
	return A_pred


def BPl_decode(Z):


	adj_pred = torch.matmul(Z, Z.t())
	adj_pred = torch.clamp(adj_pred, min=-np.Inf, max=25)

	adjpred = 1 - torch.exp(-adj_pred.exp())
	# A_pred = adjpred
	SMALL = 1e-3
	A_pred = torch.clamp(adjpred, min=SMALL, max=1 - SMALL)

	# print(torch.min(A_pred))
	# assert torch.min(A_pred)>=0
	# print(torch.max(A_pred))
	# assert torch.max(A_pred)<=1

	return A_pred


def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)



def normal_init(input_dim, output_dim):
	initial = torch.empty(input_dim, output_dim)
	return nn.Parameter(nn.init.normal_(initial, mean=0, std=0.1))



def he_init(input_dim, output_dim):
	initial = torch.empty(input_dim, output_dim)
	return nn.Parameter(nn.init.kaiming_normal_(initial))



class SparseDropout(nn.Module):
	def __init__(self, kprob):
		super(SparseDropout, self).__init__()
		self.kprob = kprob
	def forward(self,x):
		# print(x)
		mask = ((torch.rand(x._values().size())+(self.kprob)).floor()).type(torch.bool)
		# floor 向下取整
		# print('size')
		size = x.size()
		# print('mask')
		# print(mask)
		rc = x._indices()[:,mask]
		# print('rc')
		# print(rc)
		val = x._values()[mask]*(1.0/self.kprob)
		# dropout要对值也进行处理的吗？

		# VALUE 返回5是因为要使得值不变
		# print('val')
		# print(val)
		p = torch.sparse.FloatTensor(rc, val, size)
		# print(p)
		return p




class bilinear_decode(nn.Module):
	def __init__(self, z_dimension, activation = torch.sigmoid,  **kwargs):
		super(bilinear_decode, self).__init__(**kwargs)
		self.decode_weight0 = glorot_init(z_dimension, z_dimension)


	def forward(self, inputs):

		z = inputs
		x = torch.sigmoid(torch.mm(torch.mm(z, self.decode_weight0), z.t()))

		return x



class GraphConvSparse(nn.Module):
	def __init__(self, input_dim, output_dim, adj, activation = F.tanh, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		# self.weight = glorot_init(input_dim, output_dim)
		# self.weight = he_init(input_dim, output_dim)
		# self.weight = nn.Parameter(torch.Tensor(input_dim,output_dim),requires_grad=True)
		self.weight = normal_init(input_dim, output_dim)
		# self.weight = he_init(input_dim, output_dim)

		self.adj = adj
		self.activation = activation
		# self.reset_parameters
	# def reset_parameters(self):
	# 	self.weight.data.normal_((0,1))

	def forward(self, inputs):
		x = inputs
		x = torch.mm(x, self.weight)
		x = torch.mm(self.adj, x)
		outputs = self.activation(x)
		return outputs
		# return x



class GraphConvSparse_(nn.Module):
	def __init__(self, input_dim, output_dim, activation = F.tanh, **kwargs):
		super(GraphConvSparse_, self).__init__(**kwargs)

		self.weight = he_init(input_dim, output_dim)
		self.activation = activation
		# self.reset_parameters
	# def reset_parameters(self):
	# 	self.weight.data.normal_((0,1))

	def forward(self, inputs, adj):
		x = inputs
		x = torch.mm(x, self.weight)
		x = torch.mm(adj, x)
		outputs = self.activation(x)
		return outputs




class multi_layer_gnn(nn.Module):
	def __init__(self,adj,input_dim,hidden2_dim,hidden3_dim,hidden4_dim):
		super(multi_layer_gnn, self).__init__()
		self.base_gcn = GraphConvSparse(input_dim, hidden2_dim, adj)
		self.out_gcn = GraphConvSparse(hidden2_dim, hidden3_dim, adj)
		self.final_gcn = GraphConvSparse(hidden3_dim, hidden4_dim, adj, activation=lambda x:x)
		# self.bn0 = nn.BatchNorm1d(hidden2_dim)
		# self.bn1 = nn.BatchNorm1d(hidden3_dim)

	def forward(self,x):
		hidden = self.base_gcn(x)
		# hidden = self.bn0(hidden)
		mid = self.out_gcn(hidden)
		# mid = self.bn1(mid)
		final = self.final_gcn(mid)
		# z = torch.mean(torch.stack((hidden,mid,final),dim=0),dim=0)
		z = torch.max(torch.stack((hidden,mid,final),dim=0),dim=0)[0]

		return z




##
