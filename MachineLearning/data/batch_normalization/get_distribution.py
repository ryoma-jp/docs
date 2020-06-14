#! -*- coding: utf-8 -*-

#---------------------------------
# モジュールのインポート
#---------------------------------
import argparse
import numpy as np
from tensorflow.keras.datasets import cifar10

#---------------------------------
# 定数定義
#---------------------------------

#---------------------------------
# 関数
#---------------------------------
def ArgParser():
	parser = argparse.ArgumentParser(description='データセットの分布(平均と標準偏差)を返す',
				formatter_class=argparse.RawTextHelpFormatter)

	# --- 引数を追加 ---
	parser.add_argument('--dataset', dest='dataset', type=str, default=None, required=True, \
			help='データセット名(\'cifar10\')')

	args = parser.parse_args()

	return args

def get_distribution_cifar10():
	# --- CIFAR-10のデータセット取得 ---
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	
#	print(x_train.shape)
#	print(np.mean(x_train / 255, axis=(0, 1, 2)))
#	print(np.std(x_train / 255, axis=(0, 1, 2)))
	
	mean = np.mean(x_train / 255, axis=(0, 1, 2))
	std = np.std(x_train / 255, axis=(0, 1, 2))
	
	return mean, std
	
def main():
	# --- 引数処理 ---
	args = ArgParser()
	
	# --- データセットの分布取得 ---
	if (args.dataset == 'cifar10'):
		mean, std = get_distribution_cifar10()
	else:
		print('[ERROR] {} is not supported'.format(args.dataset))
		quit()
	
	print('[Result]')
	print('  * mean : {}'.format(mean))
	print('  * std  : {}'.format(std))
	
	return

#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
	main()

