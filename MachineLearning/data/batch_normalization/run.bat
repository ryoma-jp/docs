@echo off

rem 学習込み
python batch_normalization.py --with_train --model_dir .\pytorch_model --model_name NN_PyTorch.pth

rem 検証のみ
rem python batch_normalization.py --model_dir .\pytorch_model --model_name NN_PyTorch.pth


