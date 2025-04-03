#!/usr/bin/env python
"""
inference.py

全 valid indices のサンプルについて、1ショット推論を実施し、
各サンプルのタイムスタンプと4クラスの尤度（例：[0.1, 0.1, 0.7, 0.1]）を返すスクリプトです。

実行例:
$ python inference.py --params params/main/params.yaml --fold 6 --data_root ./datasets --cuda_device 0 --history 4 --trial_name 090 --mode test --resume_from_checkpoint checkpoints/main/090_stage1_best.pth
"""

import sys
import json
import os
from argparse import Namespace
from utils.main.config import parse_params
from datasets.main.dataloader import load_datasets
# ExperimentManager は main.py 内に実装されているものと仮定します
from main import ExperimentManager

def main():
    # コマンドラインと YAML からパラメータを取得
    args, yaml_config = parse_params(dump=True)
    
    # 推論はCPU上で実施するため、強制的に device を "cpu" に設定
    args.device = "cpu"
    
    # ExperimentManager を初期化（モデル構築等を実施）
    experiment = ExperimentManager(args)
    
    # チェックポイントパスの指定確認
    if not args.resume_from_checkpoint:
        print("Error: 推論実行には --resume_from_checkpoint でチェックポイントパスを指定してください。")
        sys.exit(1)
    
    print(f"Loading checkpoint from: {args.resume_from_checkpoint}")
    experiment.load_checkpoint(args.resume_from_checkpoint)
    
    # モデルを CPU へ移動し、評価モードに設定
    experiment.model.to("cpu")
    experiment.model.eval()
    
    # データセット読み込み（train, valid, test の3セットが返る）
    _, _, test_dataset = load_datasets(args, debug=args.debug)
    
    # validデータセットからタイムスタンプも返すためにフラグを有効化
    test_dataset.return_timestamp = True
    
    predictions = {}
    num_samples = len(test_dataset)
    print(f"Valid dataset: {num_samples} samples found.")
    
    # valid dataset 内の全サンプルに対して推論を実施
    for idx in range(num_samples):
        sample = test_dataset[idx]
        # __getitem__ の返り値が (X, h, y, timestamp) となるはずです
        if len(sample) == 4:
            X, h, y, timestamp = sample
        else:
            print(f"Warning: Sample index {idx} did not return timestamp; skipping.")
            continue
        
        # X の shape は [history, channels, height, width]（例: [4, 10, 256, 256]）
        # モデル入力はバッチ次元付きの [1, 4, 10, 256, 256] を想定しているため、unsqueeze
        sample_input1 = X.unsqueeze(0)  # -> [1, 4, 10, 256, 256]
        sample_input2 = h.unsqueeze(0)  # -> [1, 672, 128]
        
        # 1ショット推論を実施
        probabilities = experiment.predict_one_shot(sample_input1, sample_input2)
        
        # タイムスタンプを "YYYYMMDDHH" 形式の文字列に変換
        ts_key = timestamp.strftime("%Y%m%d%H")
        predictions[ts_key] = probabilities
    
    # 推論結果を表示
    print("Inference results (timestamp and 4-class likelihoods):")
    for ts, probs in predictions.items():
        print(f"Timestamp: {ts}, Probabilities: {probs}")
    
    # ../data ディレクトリが存在しなければ作成
    pred_dir = os.path.join("..", "data")
    os.makedirs(pred_dir, exist_ok=True)
    
    # 結果を ../data/pred.json に保存
    pred_path = os.path.join(pred_dir, "pred.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Predictions saved to: {pred_path}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
