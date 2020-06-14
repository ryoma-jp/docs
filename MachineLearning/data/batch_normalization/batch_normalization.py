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
	parser.add_argument('--model_id', dest='model_id', type=int, default=0, required=False, \
			help='モデルID\n'
				 '  model_id  description\n'
				 '  0         LeNet\n'
				 '  1         LeNet with Batch Normalization')
	parser.add_argument('--enable_augmentation', dest='enable_augmentation', action='store_true', required=False, \
			help='Data Augmentationを有効にする')
	
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
		out = self.pool(F.relu(self.conv1(x)))
		out = self.pool(F.relu(self.conv2(out)))
		out = out.view(-1, 16 * 5 * 5)
		out = F.relu(self.fc1(out))
		out = F.relu(self.fc2(out))
		out = self.fc3(out)
		return out
	
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
		out = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
		out = self.pool(F.relu(self.conv2_bn(self.conv2(out))))
		out = out.view(-1, 16 * 5 * 5)
		out = F.relu(self.fc1_bn(self.fc1(out)))
		out = F.relu(self.fc2_bn(self.fc2(out)))
		out = self.fc3(out)
		return out
	
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
	
	def __init__(self, dataset=DATASET_CIFAR10, model_id=0, enable_augmentation=False):
		# --- ハイパーパラメータ ---
		self.batch_size = 32
		self.n_epoch = 100
		
		# --- データロード ---
		self.trainloader, self.testloader, self.classes = self._load_dataset(dataset, enable_augmentation=enable_augmentation)
		
		# --- モデル構築 ---
		print('[INFO] model_id={}'.format(model_id))
		if (model_id == 0):
			self.net = NN_PyTorch_Net_00()
		else:
			self.net = NN_PyTorch_Net_01_BN()
		
		# --- GPU設定 ---
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
	
	def _load_dataset(self, dataset, enable_augmentation=False):
		if (dataset == self.DATASET_CIFAR10):
			if (enable_augmentation):
				mean = (0.49139968, 0.48215841, 0.44653091)
				std = (0.24703223, 0.24348513, 0.26158784)
			
				transform = transforms.Compose([
					transforms.RandomCrop(32, padding=4),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					transforms.Normalize(mean, std)])
			
			else:
				transform = transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			
			trainset = torchvision.datasets.CIFAR10(root='data_cifar10', train=True,
							download=True, transform=transform)
			trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
							shuffle=True, num_workers=2)
			
			testset = torchvision.datasets.CIFAR10(root='data_cifar10', train=False,
							download=True, transform=transform)
			testloader = torch.utils.data.DataLoader(testset, batch_size=100,
							shuffle=False, num_workers=2)
			
			classes = ('plane', 'car', 'bird', 'cat',
						'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
			
#			dataiter = iter(trainloader)
#			images, labels = dataiter.next()
#			
#			self._imshow(torchvision.utils.make_grid(images))
#			print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
		
		return trainloader, testloader, classes
	
	def _calc_accuracy(self, dataloader):
		correct = 0
		total = 0
		with torch.no_grad():
			for data in dataloader:
				images, labels = data
				outputs = self.net(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
				
		accuracy = 100 * correct / total
		
		return accuracy
	
	def fit(self, model_dir=MODEL_DIR, model_name=MODEL_NAME):
		if (model_dir is None):
			model_dir = self.MODEL_DIR
		if (model_name is None):
			model_dir = self.MODEL_NAME
		
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
		
		log_period = int(len(self.trainloader.dataset) / self.batch_size / 5)
		loss_log = []
		loss_log_header = ['epoch', 'iter', 'loss', 'train_acc', 'test_acc']
		print('epoch, iter, loss, train_acc, test_acc')
		for epoch in range(self.n_epoch):
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
				if (i % log_period == log_period-1):
					train_acc = self._calc_accuracy(self.trainloader)
					test_acc = self._calc_accuracy(self.testloader)
					print('%3d, %5d, %.3f, %.2f, %.2f' %
						(epoch + 1, int(i / log_period) + 1, running_loss / log_period, train_acc, test_acc))
					loss_log.append([epoch + 1, int(i / log_period) + 1, running_loss / log_period, train_acc, test_acc])
					running_loss = 0.0
		
		print('Finished Training')
		
		saved_model = os.path.join(model_dir, model_name)
		os.makedirs(model_dir, exist_ok=True)
		torch.save(self.net.state_dict(), saved_model)
		
		pd.DataFrame(loss_log).to_csv(os.path.join(model_dir, 'loss_log.csv'), header=loss_log_header, index=None)
		
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
		
		accuracy = self._calc_accuracy(self.testloader)
		print('Accuracy of the network on the 10000 test images: %d %%' % (accuracy))
		
		return accuracy
	
	def get_weights(self):
		return self.net.get_weights()
	
def main():
	# --- 引数処理 ---
	args = ArgParser()
	
	# --- 学習 ---
	nn_pytorch = NN_PyTorch(model_id=args.model_id, enable_augmentation=args.enable_augmentation)
	
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

