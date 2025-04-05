#!/usr/bin/env python
"""
new_inference.py

datasets/all_data_hours/ に存在する h5 ファイル群から、
「推論可能なすべての時刻」について、連続する history 個（例: 4個）の h5 ファイルを用いて推論を行います。

具体的には、各ターゲット時刻 T について、
T-6h, T-4h, T-2h, T の4つの h5 ファイルが存在すれば、その時刻 T に対する推論を実施します。
推論結果は、最新の時刻 T の "YYYYMMDDHH" 形式をキーとして、4クラスの尤度（例：[0.1, 0.1, 0.7, 0.1]）を値とし、
最終的に "../data/pred.json" に保存されます.

実行例:
$ python new_inference.py --params params/main/params.yaml --fold 6 --data_root ./datasets --cuda_device 0 --history 4 --trial_name 090 --mode test --resume_from_checkpoint checkpoints/main/090_stage1_best.pth
"""

import os
import sys
import json
import h5py
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from argparse import ArgumentParser, Namespace
from utils.main.config import parse_params
from models.main.models.ours import Ours

def load_checkpoint(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu")
    # チェックポイントが "model_state_dict" を持っている場合
    if "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state

    # "total_ops" や "total_params" を含むキーを除外
    filtered_state_dict = {k: v for k, v in state_dict.items() if "total_ops" not in k and "total_params" not in k}

    model.load_state_dict(filtered_state_dict)
    return model

def namespace_to_dict(obj):
    """再帰的に Namespace や list を辞書に変換する"""
    if isinstance(obj, Namespace):
        return {k: namespace_to_dict(getattr(obj, k)) for k in obj.__dict__}
    elif isinstance(obj, dict):
        return {k: namespace_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [namespace_to_dict(v) for v in obj]
    else:
        return obj

def main():
    parser = ArgumentParser()
    parser.add_argument("--params", required=True, help="YAML設定ファイルのパス")
    parser.add_argument("--fold", type=int, required=True, help="Fold番号")
    parser.add_argument("--data_root", default="./datasets", help="datasetsディレクトリのルート")
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--history", type=int, default=4, help="使用する履歴ファイル数（例: 4）")
    parser.add_argument("--trial_name", default="idxxxx")
    parser.add_argument("--mode", default="test")
    parser.add_argument("--resume_from_checkpoint", required=True, help="チェックポイントのパス")
    args = parser.parse_args()

    # 既存の設定ファイルからパラメータを読み込む
    args_config, yaml_config = parse_params(dump=True)
    # コマンドラインの各パラメータで上書き
    args_config.fold = args.fold
    args_config.data_root = args.data_root
    args_config.cuda_device = args.cuda_device
    args_config.history = args.history
    args_config.trial_name = args.trial_name
    args_config.mode = args.mode
    args_config.resume_from_checkpoint = args.resume_from_checkpoint
    # 推論はCPU上で実施する
    args_config.device = "cpu"
    # h5ファイルの格納先は all_data_hours 以下
    args_config.data_path = os.path.join(args_config.data_root, "all_data_hours")

    # キャッシュされた統計情報（means.npy, stds.npy）を読み込む
    fold_dir = os.path.join(args_config.cache_root, f"fold{args_config.fold}")
    train_cache_dir = os.path.join(fold_dir, "train")
    means_path = os.path.join(train_cache_dir, "means.npy")
    stds_path = os.path.join(train_cache_dir, "stds.npy")
    if not os.path.exists(means_path) or not os.path.exists(stds_path):
        print("Error: Cached statistics (means.npy or stds.npy) not found.")
        sys.exit(1)
    means = np.load(means_path)   # shape: (10,)
    stds = np.load(stds_path)       # shape: (10,)

    # 全ての h5 ファイル一覧を取得し、タイムスタンプでソート
    all_files = [f for f in os.listdir(args_config.data_path) if f.endswith(".h5")]
    file_dict = {}
    for f in all_files:
        # ファイル名例: "20250401_000000.h5"
        try:
            ts_str = os.path.splitext(f)[0]
            ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            file_dict[ts] = os.path.join(args_config.data_path, f)
        except Exception as e:
            print(f"Warning: ファイル名のパースに失敗: {f} ({e})")
            continue

    if not file_dict:
        print("Error: 有効な h5 ファイルが見つかりませんでした。")
        sys.exit(1)

    sorted_times = sorted(file_dict.keys())

    # 推論可能なターゲット時刻を抽出
    # 各ターゲット時刻 T に対して、T, T-2h, T-4h, T-6h が存在すれば対象とする
    valid_targets = []
    for t in sorted_times:
        required = [t - timedelta(hours=2*i) for i in range(0, args_config.history)]
        if all(rt in file_dict for rt in required):
            valid_targets.append(t)

    if not valid_targets:
        print("Error: 推論可能なターゲット時刻が見つかりませんでした。")
        sys.exit(1)

    print("推論可能なターゲット時刻（最新時刻）一覧:")
    for vt in valid_targets:
        print(vt.strftime("%Y%m%d_%H%M%S"))

    # モデルのインスタンス化（config の設定から architecture_params を使用）
    arch_params = args_config.model.models[args_config.model.selected].architecture_params
    params_dict = namespace_to_dict(arch_params)
    model = Ours(**params_dict)
    model.to("cpu")
    model.eval()
    # チェックポイントからモデルパラメータをロード
    load_checkpoint(model, args_config.resume_from_checkpoint)

    predictions = {}
    # 各ターゲット時刻について推論を実施
    for target in valid_targets:
        # required_times は [T-6h, T-4h, T-2h, T] の昇順のリスト
        required_times = [target - timedelta(hours=2*(i)) for i in range(args_config.history)]
        X_list = []
        for rt in required_times:
            file_path = file_dict.get(rt)
            if file_path is None:
                print(f"Warning: {rt.strftime('%Y%m%d_%H%M%S')} のファイルが存在しません。スキップします。")
                X_list = []
                break
            with h5py.File(file_path, "r") as f:
                X_data = f["X"][:]  # shape: (10,256,256) を想定
                X_list.append(X_data)
        if not X_list or len(X_list) != args_config.history:
            continue
        # スタックして形状 (history, 10, 256,256) にする
        X_array = np.stack(X_list, axis=0)
        X_array = np.nan_to_num(X_array, 0)
        # 正規化：各チャネル毎に (x - mean) / (std + 1e-8)
        X_norm = (X_array - means[None, :, None, None]) / (stds[None, :, None, None] + 1e-8)
        # テンソルに変換し、バッチ次元を追加 → (1, history, 10,256,256)
        sample_input1 = torch.from_numpy(X_norm).float().unsqueeze(0)
        # 特徴量入力はゼロテンソル（(1,672,128)）
        sample_input2 = torch.zeros((1, 672, 128), dtype=torch.float32)
        
        with torch.no_grad():
            logits, _ = model(sample_input1, sample_input2)
            probs = torch.softmax(logits, dim=1)
            probs_list = probs[0].cpu().numpy().tolist()
        
        # キーはターゲット時刻（最新）の "YYYYMMDDHH" 形式
        key = target.strftime("%Y%m%d%H")
        predictions[key] = [round(p, 6) for p in probs_list]

    # 結果を "../data/pred.json" に保存
    out_dir = os.path.join("..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pred.json")
    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Saved predictions for {len(predictions)} timestamps to {out_path}")
    print("Prediction results:")
    for ts, probs in predictions.items():
        print(f"{ts}: {probs}")

if __name__ == "__main__":
    main()
