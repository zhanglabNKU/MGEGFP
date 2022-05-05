import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from layer import *

import args



class MGEGFP(nn.Module):
	def __init__(self, adj_all, nview):
		super(MGEGFP, self).__init__()
		self.nview = nview
		self.adj_0 = adj_all[0]
		self.adj_1 = adj_all[1]
		self.adj_2 = adj_all[2]
		self.adj_3 = adj_all[3]
		self.adj_4 = adj_all[4]
		self.adj_5 = adj_all[5]
		# self.base_gcns0 = GraphConvSparse(args.input_dim, args.hidden2_dim, adj0, activation=lambda x:x)
		# self.base_gcns1 = GraphConvSparse(args.input_dim, args.hidden2_dim, adj1, activation=lambda x:x)
		# self.base_gcns2 = GraphConvSparse(args.input_dim, args.hidden2_dim, adj2, activation=lambda x:x)
		# self.base_gcns3 = GraphConvSparse(args.input_dim, args.hidden2_dim, adj3, activation=lambda x:x)
		# self.base_gcns4 = GraphConvSparse(args.input_dim, args.hidden2_dim, adj4, activation=lambda x:x)
		# self.base_gcns5 = GraphConvSparse(args.input_dim, args.hidden2_dim, adj5, activation=lambda x:x)

		self.base_gcns0 = GraphConvSparse(args.input_dim, args.hidden_dim, adj_all[0])
		self.base_gcns1 = GraphConvSparse(args.input_dim, args.hidden_dim, adj_all[1])
		self.base_gcns2 = GraphConvSparse(args.input_dim, args.hidden_dim, adj_all[2])
		self.base_gcns3 = GraphConvSparse(args.input_dim, args.hidden_dim, adj_all[3])
		self.base_gcns4 = GraphConvSparse(args.input_dim, args.hidden_dim, adj_all[4])
		self.base_gcns5 = GraphConvSparse(args.input_dim, args.hidden_dim, adj_all[5])


		self.out_gcns0 = GraphConvSparse(args.hidden_dim, args.hidden_dim, adj_all[0])
		self.out_gcns1 = GraphConvSparse(args.hidden_dim, args.hidden_dim, adj_all[1])
		self.out_gcns2 = GraphConvSparse(args.hidden_dim, args.hidden_dim, adj_all[2])
		self.out_gcns3 = GraphConvSparse(args.hidden_dim, args.hidden_dim, adj_all[3])
		self.out_gcns4 = GraphConvSparse(args.hidden_dim, args.hidden_dim, adj_all[4])
		self.out_gcns5 = GraphConvSparse(args.hidden_dim, args.hidden_dim, adj_all[5])


		self.final_gcns0 = GraphConvSparse(args.hidden_dim, args.hidden_dim, adj_all[0], activation=lambda x:x)
		self.final_gcns1 = GraphConvSparse(args.hidden_dim, args.hidden_dim, adj_all[1], activation=lambda x:x)
		self.final_gcns2 = GraphConvSparse(args.hidden_dim, args.hidden_dim, adj_all[2], activation=lambda x:x)
		self.final_gcns3 = GraphConvSparse(args.hidden_dim, args.hidden_dim, adj_all[3], activation=lambda x:x)
		self.final_gcns4 = GraphConvSparse(args.hidden_dim, args.hidden_dim, adj_all[4], activation=lambda x:x)
		self.final_gcns5 = GraphConvSparse(args.hidden_dim, args.hidden_dim, adj_all[5], activation=lambda x:x)


		self.common_gcn0 = GraphConvSparse_(args.input_dim, args.common_dim)
		self.common_gcn1 = GraphConvSparse_(args.common_dim, args.common_dim)
		self.common_gcn2 = GraphConvSparse_(args.common_dim, args.common_dim)



		self.gate0 = nn.ModuleList()
		self.gate1 = nn.ModuleList()
		self.gate2 = nn.ModuleList()
		self.gate3 = nn.ModuleList()
		self.gate4 = nn.ModuleList()
		self.gate5 = nn.ModuleList()

		for i in range(6):
			self.gate0.append(nn.Linear(args.hidden_dim + args.common_dim, 1, bias=True))
			self.gate1.append(nn.Linear(args.hidden_dim + args.common_dim, 1, bias=True))
			self.gate2.append(nn.Linear(args.hidden_dim + args.common_dim, 1, bias=True))
			self.gate3.append(nn.Linear(args.hidden_dim + args.common_dim, 1, bias=True))
			self.gate4.append(nn.Linear(args.hidden_dim + args.common_dim, 1, bias=True))
			self.gate5.append(nn.Linear(args.hidden_dim + args.common_dim, 1, bias=True))




	def forward(self, x_all,device):


		hidden0 = self.base_gcns0(x_all[0])
		hidden1 = self.base_gcns1(x_all[1])
		hidden2 = self.base_gcns2(x_all[2])
		hidden3 = self.base_gcns3(x_all[3])
		hidden4 = self.base_gcns4(x_all[4])
		hidden5 = self.base_gcns5(x_all[5])

		mid0 = self.out_gcns0(hidden0)
		mid1 = self.out_gcns1(hidden1)
		mid2 = self.out_gcns2(hidden2)
		mid3 = self.out_gcns3(hidden3)
		mid4 = self.out_gcns4(hidden4)
		mid5 = self.out_gcns5(hidden5)

		final0 = self.final_gcns0(mid0)
		final1 = self.final_gcns1(mid1)
		final2 = self.final_gcns2(mid2)
		final3 = self.final_gcns3(mid3)
		final4 = self.final_gcns4(mid4)
		final5 = self.final_gcns5(mid5)

		if args.layer_agg == 'mean':
			z_0 = torch.mean(torch.stack((hidden0,mid0,final0),dim=0),dim=0)
			z_1 = torch.mean(torch.stack((hidden1,mid1,final1),dim=0),dim=0)
			z_2 = torch.mean(torch.stack((hidden2,mid2,final2),dim=0),dim=0)
			z_3 = torch.mean(torch.stack((hidden3,mid3,final3),dim=0),dim=0)
			z_4 = torch.mean(torch.stack((hidden4,mid4,final4),dim=0),dim=0)
			z_5 = torch.mean(torch.stack((hidden5,mid5,final5),dim=0),dim=0)

		elif args.layer_agg == 'max':
			z_0 = torch.max(torch.stack((hidden0,mid0,final0),dim=0),dim=0)[0]
			z_1 = torch.max(torch.stack((hidden1,mid1,final1),dim=0),dim=0)[0]
			z_2 = torch.max(torch.stack((hidden2,mid2,final2),dim=0),dim=0)[0]
			z_3 = torch.max(torch.stack((hidden3,mid3,final3),dim=0),dim=0)[0]
			z_4 = torch.max(torch.stack((hidden4,mid4,final4),dim=0),dim=0)[0]
			z_5 = torch.max(torch.stack((hidden5,mid5,final5),dim=0),dim=0)[0]

		elif args.layer_agg == 'none':
			z_0 = final0
			z_1 = final1
			z_2 = final2
			z_3 = final3
			z_4 = final4
			z_5 = final5

		elif args.layer_agg == 'concat':
			z_0 = torch.cat([hidden0,mid0,final0],dim=1)
			z_1 = torch.cat([hidden1,mid1,final1],dim=1)
			z_2 = torch.cat([hidden2,mid2,final2],dim=1)
			z_3 = torch.cat([hidden3,mid3,final3],dim=1)
			z_4 = torch.cat([hidden4,mid4,final4],dim=1)
			z_5 = torch.cat([hidden5,mid5,final5],dim=1)

		else:
			print('wrong layer aggregation type')






		z_common0 = self.common_gcn0(x_all[0],self.adj_0)
		z_common0 = self.common_gcn1(z_common0,self.adj_0)
		z_common0 = self.common_gcn2(z_common0,self.adj_0)

		z_common1 = self.common_gcn0(x_all[1],self.adj_1)
		z_common1 = self.common_gcn1(z_common1,self.adj_1)
		z_common1 = self.common_gcn2(z_common1,self.adj_1)

		z_common2 = self.common_gcn0(x_all[2],self.adj_2)
		z_common2 = self.common_gcn1(z_common2,self.adj_2)
		z_common2 = self.common_gcn2(z_common2,self.adj_2)

		z_common3 = self.common_gcn0(x_all[3],self.adj_3)
		z_common3 = self.common_gcn1(z_common3,self.adj_3)
		z_common3 = self.common_gcn2(z_common3,self.adj_3)

		z_common4 = self.common_gcn0(x_all[4],self.adj_4)
		z_common4 = self.common_gcn1(z_common4,self.adj_4)
		z_common4 = self.common_gcn2(z_common4,self.adj_4)

		z_common5 = self.common_gcn0(x_all[5],self.adj_5)
		z_common5 = self.common_gcn1(z_common5,self.adj_5)
		z_common5 = self.common_gcn2(z_common5,self.adj_5)

		z0 = torch.cat([z_common0, z_0],dim=1)
		z1 = torch.cat([z_common1, z_1],dim=1)
		z2 = torch.cat([z_common2, z_2],dim=1)
		z3 = torch.cat([z_common3, z_3],dim=1)
		z4 = torch.cat([z_common4, z_4],dim=1)
		z5 = torch.cat([z_common5, z_5],dim=1)


		# z0 = self.base_gcns0(x0)
		# z1 = self.base_gcns1(x1)
		# z2 = self.base_gcns2(x2)
		# z3 = self.base_gcns3(x3)
		# z4 = self.base_gcns4(x4)
		# z5 = self.base_gcns5(x5)

		z = [z0,z1,z2,z3,z4,z5]

		score0 = torch.empty(6400, 6).to(device)
		score1 = torch.empty(6400, 6).to(device)
		score2 = torch.empty(6400, 6).to(device)
		score3 = torch.empty(6400, 6).to(device)
		score4 = torch.empty(6400, 6).to(device)
		score5 = torch.empty(6400, 6).to(device)

		for i in range(6):
			score0[:,i] = self.gate0[i](z[i]).squeeze()
			score1[:,i] = self.gate1[i](z[i]).squeeze()
			score2[:,i] = self.gate2[i](z[i]).squeeze()
			score3[:,i] = self.gate3[i](z[i]).squeeze()
			score4[:,i] = self.gate4[i](z[i]).squeeze()
			score5[:,i] = self.gate5[i](z[i]).squeeze()



		score0 = torch.softmax(score0,dim=1)
		score1 = torch.softmax(score1,dim=1)
		score2 = torch.softmax(score2,dim=1)
		score3 = torch.softmax(score3,dim=1)
		score4 = torch.softmax(score4,dim=1)
		score5 = torch.softmax(score5,dim=1)


		z_global_0 = (score0[:,0].view(-1, 1))*z0 + (score0[:,1].view(-1, 1))*z1 + (score0[:,2].view(-1, 1))*z2 + (score0[:,3].view(-1, 1))*z3 + (score0[:,4].view(-1, 1))*z4 + (score0[:,5].view(-1, 1))*z5
		z_global_1 = (score1[:,0].view(-1, 1))*z0 + (score1[:,1].view(-1, 1))*z1 + (score1[:,2].view(-1, 1))*z2 + (score1[:,3].view(-1, 1))*z3 + (score1[:,4].view(-1, 1))*z4 + (score1[:,5].view(-1, 1))*z5
		z_global_2 = (score2[:,0].view(-1, 1))*z0 + (score2[:,1].view(-1, 1))*z1 + (score2[:,2].view(-1, 1))*z2 + (score2[:,3].view(-1, 1))*z3 + (score2[:,4].view(-1, 1))*z4 + (score2[:,5].view(-1, 1))*z5
		z_global_3 = (score3[:,0].view(-1, 1))*z0 + (score3[:,1].view(-1, 1))*z1 + (score3[:,2].view(-1, 1))*z2 + (score3[:,3].view(-1, 1))*z3 + (score3[:,4].view(-1, 1))*z4 + (score3[:,5].view(-1, 1))*z5
		z_global_4 = (score4[:,0].view(-1, 1))*z0 + (score4[:,1].view(-1, 1))*z1 + (score4[:,2].view(-1, 1))*z2 + (score4[:,3].view(-1, 1))*z3 + (score4[:,4].view(-1, 1))*z4 + (score4[:,5].view(-1, 1))*z5
		z_global_5 = (score5[:,0].view(-1, 1))*z0 + (score5[:,1].view(-1, 1))*z1 + (score5[:,2].view(-1, 1))*z2 + (score5[:,3].view(-1, 1))*z3 + (score5[:,4].view(-1, 1))*z4 + (score5[:,5].view(-1, 1))*z5




		assert not torch.any(torch.isnan(z0))
		assert not torch.any(torch.isnan(z1))
		assert not torch.any(torch.isnan(z2))
		assert not torch.any(torch.isnan(z3))
		assert not torch.any(torch.isnan(z4))
		assert not torch.any(torch.isnan(z5))




		z_global=[z_global_0,z_global_1,z_global_2,z_global_3,z_global_4,z_global_5]
		z_out = [z0, z1, z2, z3, z4, z5]

		return z_global,z_out





##
