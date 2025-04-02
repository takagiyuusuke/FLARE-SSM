#!/usr/bin/env python
"""
inference.py

1ショット推論をCPU上で実施するためのスクリプトです。
コマンドライン引数やYAML設定ファイルからパラメータを読み込み、
学習済みのチェックポイントをロードし、テストデータから1サンプルを取得して推論します。

推論結果は4クラスの尤度（例：[0.1, 0.1, 0.7, 0.1]）として出力されます。
"""

import sys
import torch
import os
from argparse import Namespace
from utils.main.config import parse_params
from datasets.main.dataloader import load_datasets
# ExperimentManagerはmain.py内に実装されているものとします
from main import ExperimentManager

def main():
    # コマンドラインとYAMLからパラメータを取得
    args, yaml_config = parse_params(dump=True)
    
    # 推論はCPUで行うため、明示的にデバイスをCPUに設定
    args.device = "cpu"
    
    # ExperimentManagerを生成（これによりモデル構築などが行われる）
    experiment = ExperimentManager(args)
    
    # チェックポイントのパスが指定されているか確認
    if not args.resume_from_checkpoint:
        print("Error: 推論実行には --resume_from_checkpoint でチェックポイントパスを指定してください。")
        sys.exit(1)
    
    print(f"Loading checkpoint from: {args.resume_from_checkpoint}")
    experiment.load_checkpoint(args.resume_from_checkpoint)
    
    # モデルをCPUへ移動し、評価モードに設定
    experiment.model.to("cpu")
    experiment.model.eval()
    
    # テストデータセットを読み込み（load_datasetsは train, valid, test の3セットを返す）
    _, _, test_dataset = load_datasets(args, debug=args.debug)
    
    # テストデータセットから1サンプルを取得（ここでは先頭のサンプルを利用）
    # dataset.__getitem__の返り値は (X, h, y) です
    sample = test_dataset[0]
    X, h, y = sample

    # Xのshapeは [history, channels, height, width] (例: [4, 10, 256, 256])
    # モデルのdummy_input1は [1, 4, 10, 256, 256] を前提としているため、
    # Xの次元を調整（batch次元を追加）
    sample_input1 = X.unsqueeze(0)  # [1, 4, 10, 256, 256]
    sample_input2 = h.unsqueeze(0)  # [1, 672, 128]
    
    # 1ショット推論実施
    probabilities = experiment.predict_one_shot(sample_input1, sample_input2)
    
    print("1-shot inference probabilities (4-class likelihoods):")
    print(probabilities)

if __name__ == "__main__":
    main()
