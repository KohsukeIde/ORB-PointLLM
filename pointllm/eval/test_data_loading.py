#!/usr/bin/env python3
"""
ModelNet40 データ読み込みテストスクリプト
マルチクラウド評価の前にデータが正しく読み込めるかテスト
"""

import os
import sys
import torch
import numpy as np
from pointllm.data import ModelNet

def test_modelnet_loading():
    """ModelNet40データの読み込みテスト"""
    
    data_path = "/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat"
    
    print("=== ModelNet40 Data Loading Test ===")
    print(f"Data path: {data_path}")
    
    # ファイル存在確認
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found: {data_path}")
        return False
    
    file_size = os.path.getsize(data_path) / (1024 * 1024)  # MB
    print(f"✅ File exists: {file_size:.1f} MB")
    
    try:
        print("\n--- Loading dataset (subset of 10 samples) ---")
        dataset = ModelNet(
            config_path=None,
            split="test", 
            subset_nums=10,
            use_color=False,
            data_path=data_path
        )
        
        print(f"✅ Dataset loaded successfully")
        print(f"   Total samples: {len(dataset)}")
        
        # 最初のサンプルをテスト
        print("\n--- Testing first sample ---")
        sample = dataset[0]
        
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Point cloud shape: {sample['point_clouds'].shape}")
        print(f"Point cloud dtype: {sample['point_clouds'].dtype}")
        print(f"Label: {sample['labels']} ({sample['label_names']})")
        print(f"Index: {sample['indice']}")
        
        # 統計情報
        points = sample['point_clouds'].numpy()
        print(f"\nPoint cloud statistics:")
        print(f"  Min: {points.min():.3f}")
        print(f"  Max: {points.max():.3f}")
        print(f"  Mean: {points.mean(axis=0)}")
        print(f"  Std: {points.std(axis=0)}")
        
        # 複数サンプルのテスト
        print("\n--- Testing multiple samples ---")
        labels = []
        label_names = []
        
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            labels.append(sample['labels'])
            label_names.append(sample['label_names'])
        
        print(f"Sample labels: {labels}")
        print(f"Sample names: {label_names}")
        print(f"Unique categories in sample: {len(set(labels))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_cloud_simulation():
    """マルチクラウドサンプリングのテスト"""
    
    data_path = "/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat"
    
    print("\n=== Multi-Cloud Sampling Test ===")
    
    try:
        dataset = ModelNet(
            config_path=None,
            split="test",
            subset_nums=50,  # 小さなサブセットでテスト
            use_color=False,
            data_path=data_path
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # 2点群サンプリング
        print("\n--- 2-cloud sampling test ---")
        import random
        random.seed(42)
        
        for test_case in range(3):
            indices = random.sample(range(len(dataset)), 2)
            samples = [dataset[idx] for idx in indices]
            
            labels = [s['labels'] for s in samples]
            names = [s['label_names'] for s in samples]
            
            same_category = labels[0] == labels[1]
            
            print(f"Test case {test_case + 1}:")
            print(f"  Indices: {indices}")
            print(f"  Labels: {labels}")
            print(f"  Names: {names}")
            print(f"  Same category: {same_category}")
            
        # 3点群サンプリング
        print("\n--- 3-cloud sampling test ---")
        indices = random.sample(range(len(dataset)), 3)
        samples = [dataset[idx] for idx in indices]
        
        labels = [s['labels'] for s in samples]
        names = [s['label_names'] for s in samples]
        
        print(f"3-cloud test:")
        print(f"  Indices: {indices}")
        print(f"  Labels: {labels}")
        print(f"  Names: {names}")
        print(f"  Unique categories: {len(set(labels))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in multi-cloud test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン関数"""
    print("Starting ModelNet40 data loading tests...")
    
    # データ読み込みテスト
    success1 = test_modelnet_loading()
    
    if success1:
        # マルチクラウドサンプリングテスト
        success2 = test_multi_cloud_simulation()
    else:
        success2 = False
    
    print("\n=== Test Summary ===")
    print(f"Data loading test: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Multi-cloud test: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! Ready for multi-cloud evaluation.")
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 