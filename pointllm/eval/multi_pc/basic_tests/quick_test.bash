#!/bin/bash

# Quick Test Script for Multi-Cloud Evaluation
# 動作確認用の軽量テスト

# CUDAモジュールの読み込み（GPU使用可能にする）
module load cuda/12.6 2>/dev/null || echo "CUDA module not available, using CPU"

DATA_PATH="/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat"
MODEL_NAME="RunsenXu/PointLLM_7B_v1.2"
OUTPUT_DIR="/groups/gag51404/ide/ORB-PointLLM/pointllm/eval/multi_pc/quick_test_results"

echo "=== Quick Multi-Cloud Test (ORB-PointLLM) ==="
echo "Data: $DATA_PATH"
echo "Model: $MODEL_NAME"
echo "CUDA: $(which nvcc 2>/dev/null && echo 'Available' || echo 'Not available - using CPU')"

# データが存在するか確認
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

echo "Data file found: $(ls -lh $DATA_PATH | awk '{print $5}')"

# 1. 小規模なShape Matching テスト (2点群、10サンプル - 同じ・異なるカテゴリ両方を含む)
echo ""
echo "1. Quick Shape Matching Test (2 clouds, 10 samples - mixed same/different categories) with Multi-Cloud ENABLED..."
python ../core/eval_multi_cloud.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --task_type shape_matching \
    --num_clouds 2 \
    --num_samples 10 \
    --subset_nums 100 \
    --use_color \
    --pointnum 8192 \
    --output_dir "$OUTPUT_DIR/quick_test" \
    --force_enable_multi_cloud \
    --max_point_clouds 8 \
    --force_regenerate

# 2. 単一点群テスト (比較用)
echo ""
echo "2. Single Cloud Test (5 samples)..."
python ../core/eval_multi_cloud.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --task_type object_identification \
    --num_clouds 1 \
    --num_samples 5 \
    --subset_nums 50 \
    --use_color \
    --pointnum 8192 \
    --output_dir "$OUTPUT_DIR/single_test" \
    --force_regenerate

echo ""
echo "=== Quick Test Complete ==="
echo "Check results in: $OUTPUT_DIR"

# 結果の表示
if [ -d "$OUTPUT_DIR" ]; then
    echo ""
    echo "Generated files:"
    find "$OUTPUT_DIR" -name "*.json" -exec echo "  {}" \; 
    
    echo ""
    echo "Multi-cloud configuration summary:"
    for file in $(find "$OUTPUT_DIR" -name "*.json"); do
        if [ -f "$file" ]; then
            echo "File: $(basename $file)"
            echo "  Multi-cloud enabled: $(jq -r '.model_config.enable_multi_cloud' $file 2>/dev/null || echo 'N/A')"
            echo "  Max point clouds: $(jq -r '.model_config.max_point_clouds' $file 2>/dev/null || echo 'N/A')"
            echo "  Point token length: $(jq -r '.model_config.point_token_len' $file 2>/dev/null || echo 'N/A')"
            echo ""
        fi
    done
fi 