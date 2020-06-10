@echo off

rem 学習込み
python batch_normalization.py --with_train --model_dir .\pytorch_model --model_name NN_PyTorch.pth
python batch_normalization.py --with_train --model_dir .\pytorch_model_bn --model_name NN_PyTorch.pth

rem 検証のみ
rem python batch_normalization.py --model_dir .\pytorch_model --model_name NN_PyTorch.pth

rem ヒストグラム差分
python hist_diff.py --csv1 pytorch_model\weight_conv1.csv --csv2 pytorch_model_bn\weight_conv1.csv --output hist_weight_conv1.png
python hist_diff.py --csv1 pytorch_model\weight_conv2.csv --csv2 pytorch_model_bn\weight_conv2.csv --output hist_weight_conv2.png
python hist_diff.py --csv1 pytorch_model\weight_fc1.csv --csv2 pytorch_model_bn\weight_fc1.csv --output weight_fc1.png
python hist_diff.py --csv1 pytorch_model\weight_fc2.csv --csv2 pytorch_model_bn\weight_fc2.csv --output weight_fc2.png
python hist_diff.py --csv1 pytorch_model\weight_fc3.csv --csv2 pytorch_model_bn\weight_fc3.csv --output weight_fc3.png
