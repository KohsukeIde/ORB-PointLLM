#!/bin/bash

# GPU-Enabled Multi-Cloud Test Script
# GPUを使用し、マルチクラウド機能を有効化したテスト

# CUDAモジュールの読み込み
module load cuda/12.6

DATA_PATH="/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat"
MODEL_NAME="RunsenXu/PointLLM_7B_v1.2"
OUTPUT_DIR="gpu_test_results"

echo "=== GPU Multi-Cloud Test ==="
echo "Data: $DATA_PATH"
echo "Model: $MODEL_NAME"
echo "CUDA Version: $(nvcc --version | grep release)"
echo "GPU Status: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"

# データが存在するか確認
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

echo "Data file found: $(ls -lh $DATA_PATH | awk '{print $5}')"

# 1. マルチクラウド対応 Shape Matching テスト (2点群、10サンプル)
echo ""
echo "1. Multi-Cloud Shape Matching Test (2 clouds, 10 samples) with GPU..."
python ../core/eval_multi_cloud.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --task_type shape_matching \
    --num_clouds 2 \
    --num_samples 10 \
    --subset_nums 100 \
    --use_color \
    --pointnum 8192 \
    --output_dir "$OUTPUT_DIR/multi_cloud_test" \
    --force_enable_multi_cloud \
    --max_point_clouds 8 \
    --force_regenerate

# 2. 比較用: シングルクラウドテスト（マルチクラウド無効）
echo ""
echo "2. Single Cloud Test (baseline comparison)..."
python ../core/eval_multi_cloud.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --task_type object_identification \
    --num_clouds 1 \
    --num_samples 5 \
    --subset_nums 50 \
    --use_color \
    --pointnum 8192 \
    --output_dir "$OUTPUT_DIR/single_cloud_test" \
    --force_regenerate

# 3. 3点群での part_assembly テスト
echo ""
echo "3. Multi-Cloud Part Assembly Test (3 clouds, 8 samples)..."
python ../core/eval_multi_cloud.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --task_type part_assembly \
    --num_clouds 3 \
    --num_samples 8 \
    --subset_nums 100 \
    --use_color \
    --pointnum 8192 \
    --output_dir "$OUTPUT_DIR/assembly_test" \
    --force_enable_multi_cloud \
    --max_point_clouds 8 \
    --force_regenerate

echo ""
echo "=== GPU Test Complete ==="
echo "Check results in: $OUTPUT_DIR"

# 結果の比較表示
if [ -d "$OUTPUT_DIR" ]; then
    echo ""
    echo "Generated files:"
    find "$OUTPUT_DIR" -name "*.json" -exec echo "  {}" \; 
    
    echo ""
    echo "Multi-cloud configuration summary:"
    for file in $(find "$OUTPUT_DIR" -name "*.json"); do
        echo "File: $(basename $file)"
        echo "  Multi-cloud enabled: $(jq -r '.model_config.enable_multi_cloud' $file 2>/dev/null || echo 'N/A')"
        echo "  Max point clouds: $(jq -r '.model_config.max_point_clouds' $file 2>/dev/null || echo 'N/A')"
        echo "  Point token length: $(jq -r '.model_config.point_token_len' $file 2>/dev/null || echo 'N/A')"
        echo ""
    done
fi 