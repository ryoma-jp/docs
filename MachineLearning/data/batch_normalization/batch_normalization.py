#! -*- coding: utf-8 -*-

#---------------------------------
# モジュールのインポート
#---------------------------------
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#---------------------------------
# 定数定義
#---------------------------------

#---------------------------------
# 関数
#---------------------------------
def ArgParser():
	parser = argparse.ArgumentParser(description='Batch Normalizationの効果確認用プログラム\n'
				'[参考]\n'
				'  * https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html', 
				formatter_class=argparse.RawTextHelpFormatter)

	# --- 引数を追加 ---
	parser.add_argument('--dataset', dest='dataset', type=str, default='cifar10', required=False, \
			help='データセット指定(\'cifar10\')')
	parser.add_argument('--model_dir', dest='model_dir', type=str, default=None, required=False, \
			help='学習済みモデルのパス')
	parser.add_argument('--model_name', dest='model_name', type=str, default=None, required=False, \
			help='学習済みモデル名')
	parser.add_argument('--with_train', dest='with_train', action='store_true', required=False, \
			help='学習時に設定')
	args = parser.parse_args()

	return args

#---------------------------------
# クラス
#---------------------------------
class NN_PyTorch_Net_00(nn.Module):
	
	def __init__(self):
		super(NN_PyTorch_Net_00, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
	
	def get_weights(self):
		weights = OrderedDict()
		weights['conv1'] = self.conv1.weight
		weights['conv2'] = self.conv2.weight
		weights['fc1'] = self.fc1.weight
		weights['fc2'] = self.fc2.weight
		weights['fc3'] = self.fc3.weight
		
		return weights
	
class NN_PyTorch_Net_01_BN(nn.Module):
	
	def __init__(self):
		super(NN_PyTorch_Net_01_BN, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.conv1_bn = nn.BatchNorm2d(6)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.conv2_bn = nn.BatchNorm2d(16)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc1_bn = nn.BatchNorm1d(120)
		self.fc2 = nn.Linear(120, 84)
		self.fc2_bn = nn.BatchNorm1d(84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
		x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1_bn(self.fc1(x)))
		x = F.relu(self.fc2_bn(self.fc2(x)))
		x = self.fc3(x)
		return x
	
	def get_weights(self):
		weights = OrderedDict()
		weights['conv1'] = self.conv1.weight
		weights['conv2'] = self.conv2.weight
		weights['fc1'] = self.fc1.weight
		weights['fc2'] = self.fc2.weight
		weights['fc3'] = self.fc3.weight
		
		return weights
	
class NN_PyTorch():
	DATASET_CIFAR10 = 'cifar10'
	MODEL_DIR = 'pytorch_model'
	MODEL_NAME = 'NN_PyTorch.pth'
	
	def __init__(self, dataset=DATASET_CIFAR10):
		self.trainloader, self.testloader, self.classes = self._load_dataset(dataset)
		self.net = NN_PyTorch_Net_00()
#		self.net = NN_PyTorch_Net_01_BN()
		
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		if (not(self.device == 'cpu')):
			main_ver, minor_ver = torch.cuda.get_device_capability(self.device)
			if ((main_ver > 4) or ((main_ver == 3) and (minor_ver >= 5))):
				self.net.to(self.device)
			else:
				print('[WARN] GPU is too old')
				self.device = 'cpu'
		
		return
		
	def _imshow(self, img):
		img = img / 2 + 0.5
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.show()
	
	def _load_dataset(self, dataset):
		if (dataset == self.DATASET_CIFAR10):
			transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			
			trainset = torchvision.datasets.CIFAR10(root='data_cifar10', train=True,
							download=True, transform=transform)
			trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
							shuffle=True, num_workers=2)
			
			testset = torchvision.datasets.CIFAR10(root='data_cifar10', train=False,
							download=True, transform=transform)
			testloader = torch.utils.data.DataLoader(testset, batch_size=4,
							shuffle=False, num_workers=2)
			
			classes = ('plane', 'car', 'bird', 'cat',
						'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
			
#			dataiter = iter(trainloader)
#			images, labels = dataiter.next()
#			
#			self._imshow(torchvision.utils.make_grid(images))
#			print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
		
		return trainloader, testloader, classes
	
	def fit(self, model_dir=MODEL_DIR, model_name=MODEL_NAME):
		if (model_dir is None):
			model_dir = self.MODEL_DIR
		if (model_name is None):
			model_dir = self.MODEL_NAME
		
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
		
		for epoch in range(2):
			running_loss = 0.0
			for i, data in enumerate(self.trainloader, 0):
				if (self.device == 'cpu'):
					inputs, labels = data
				else:
					inputs, labels = data[0].to(self.device), data[1].to(self.device)
				
				optimizer.zero_grad()
				
				outputs = self.net(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
				
				running_loss += loss.item()
				if (i % 2000 == 1999):
					print('[%d, %5d] loss: %.3f' %
						(epoch + 1, i + 1, running_loss / 2000))
					running_loss = 0.0
		
		print('Finished Training')
		
		saved_model = os.path.join(model_dir, model_name)
		os.makedirs(model_dir, exist_ok=True)
		torch.save(self.net.state_dict(), saved_model)
		
		return model_dir, model_name
	
	def test(self, model_path=None):
		dataiter = iter(self.testloader)
		images, labels = dataiter.next()
		
#		self._imshow(torchvision.utils.make_grid(images))
#		print('GroundTruth: ', ' '.join('%5s' % self.classes[labels[j]] for j in range(4)))
		
		self.net.load_state_dict(torch.load(model_path))
		outputs = self.net(images)
		
#		_, predicted = torch.max(outputs, 1)
#		print('Predicted: ', ' '.join('%5s' % self.classes[predicted[j]] for j in range(4)))
		
		correct = 0
		total = 0
		with torch.no_grad():
			for data in self.testloader:
				images, labels = data
				outputs = self.net(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
				
		print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
	
	def get_weights(self):
		return self.net.get_weights()
	
def main():
	# --- 引数処理 ---
	args = ArgParser()
	
	# --- 学習 ---
	nn_pytorch = NN_PyTorch()
	
	if (args.with_train):
		model_dir, model_name = nn_pytorch.fit(model_dir=args.model_dir, model_name=args.model_name)
		model = os.path.join(model_dir, model_name)
	else:
		if (args.model_dir is not None):
			model_dir = args.model_dir
			model_name = args.model_name
			model = os.path.join(args.model_dir, args.model_name)
		else:
			quit()
		
	nn_pytorch.test(model_path=model)
	weights = nn_pytorch.get_weights()
	
	for key in weights.keys():
		shape = weights[key].shape
		if (len(shape) > 2):
			weight_value = weights[key].detach().numpy().reshape(-1, shape[-1]*shape[-2])
		else:
			weight_value = weights[key].detach().numpy()
		
		pd.DataFrame(weight_value).to_csv(os.path.join(model_dir, 'weight_{}.csv'.format(key)), index=False, header=False)
		plt.hist(weight_value, bins=32)
		plt.savefig(os.path.join(model_dir, 'weight_{}_hist.png'.format(key)))
		plt.close()

	return

#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
	main()

