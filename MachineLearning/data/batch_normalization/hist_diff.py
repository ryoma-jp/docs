#! -*- coding: utf-8 -*-

#---------------------------------
# モジュールのインポート
#---------------------------------
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#---------------------------------
# 定数定義
#---------------------------------

#---------------------------------
# 関数
#---------------------------------
def ArgParser():
	parser = argparse.ArgumentParser(description='2つのcsvデータのヒストグラムを比較するツール',
				formatter_class=argparse.RawTextHelpFormatter)

	# --- 引数を追加 ---
	parser.add_argument('--csv1', dest='csv1', type=str, required=True, \
			help='CSVデータ1')
	parser.add_argument('--csv2', dest='csv2', type=str, required=True, \
			help='CSVデータ2')
	parser.add_argument('--output', dest='output', type=str, required=True, \
			help='出力ファイル名(png)')
	
	args = parser.parse_args()

	return args


def main():
	# --- 引数処理 ---
	args = ArgParser()
	
	# --- CSV読み込み ---
	df_csv1 = pd.read_csv(args.csv1, header=None)
	df_csv2 = pd.read_csv(args.csv2, header=None)
	
	labels = [args.csv1, args.csv2]
	plt.hist([df_csv1.values.reshape(-1), df_csv2.values.reshape(-1)], bins=32, stacked=False, label=labels)
	plt.legend()
	plt.tight_layout()
	plt.savefig(args.output)
	plt.close()
	
	return

#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
	main()

