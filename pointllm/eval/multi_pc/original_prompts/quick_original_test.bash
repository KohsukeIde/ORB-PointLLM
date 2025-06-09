#!/bin/bash

# Quick Original Prompts Test
# 数分で完了する軽量版のoriginal prompts動作確認

# CUDAモジュールの読み込み（GPU使用可能にする）
module load cuda/12.6 2>/dev/null || echo "CUDA module not available, using CPU"

DATA_PATH="/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat"
MODEL_NAME="RunsenXu/PointLLM_7B_v1.2"
OUTPUT_DIR="/groups/gag51404/ide/ORB-PointLLM/pointllm/eval/multi_pc/quick_original_results"

echo "=== Quick Original Prompts Test (5 samples each) ==="
echo "Testing: 'What is this?' vs custom prompts"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_NAME"
echo "CUDA: $(which nvcc 2>/dev/null && echo 'Available' || echo 'Not available - using CPU')"

# データが存在するか確認
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

echo "Data file found: $(ls -lh $DATA_PATH | awk '{print $5}')"

# 1. Single Cloud - Original "What is this?"
echo ""
echo "1. Single Cloud - Original 'What is this?' (5 samples)..."
python ../core/eval_multi_cloud.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --task_type object_identification \
    --num_clouds 1 \
    --num_samples 5 \
    --subset_nums 50 \
    --use_color \
    --pointnum 8192 \
    --output_dir "$OUTPUT_DIR" \
    --use_original_prompts \
    --prompt_index 0 \
    --force_regenerate

# 2. Multi-Cloud - Original style prompt
echo ""
echo "2. Multi-Cloud - Original style 'What are these?' (5 samples)..."
python ../core/eval_multi_cloud.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --task_type object_identification \
    --num_clouds 2 \
    --num_samples 5 \
    --subset_nums 50 \
    --use_color \
    --pointnum 8192 \
    --output_dir "$OUTPUT_DIR" \
    --force_enable_multi_cloud \
    --max_point_clouds 8 \
    --use_original_prompts \
    --prompt_index 2 \
    --force_regenerate

# 3. 比較用 - Custom prompt
echo ""
echo "3. Comparison - Custom prompt (5 samples)..."
python ../core/eval_multi_cloud.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --task_type object_identification \
    --num_clouds 2 \
    --num_samples 5 \
    --subset_nums 50 \
    --use_color \
    --pointnum 8192 \
    --output_dir "$OUTPUT_DIR" \
    --force_enable_multi_cloud \
    --max_point_clouds 8 \
    --prompt_index 0 \
    --force_regenerate

echo ""
echo "=== Quick Test Results ==="

# 結果を表示
if [ -d "$OUTPUT_DIR" ]; then
    echo ""
    echo "Generated files:"
    find "$OUTPUT_DIR" -name "*.json" -exec basename {} \; | sort
    
    echo ""
    echo "=== Prompt Comparison ==="
    
    for file in $(find "$OUTPUT_DIR" -name "*.json" | sort); do
        echo ""
        echo "--- $(basename $file) ---"
        echo "Task: $(jq -r '.task_type' $file 2>/dev/null)"
        echo "Prompt: $(jq -r '.prompt_template' $file 2>/dev/null)"
        echo "Multi-cloud: $(jq -r '.model_config.enable_multi_cloud' $file 2>/dev/null)"
        echo ""
        echo "Sample outputs:"
        jq -r '.results[0:3][] | "  Sample \(.sample_id): \(.ground_truth.label_names // .ground_truth.label_names) -> \(.model_output[0:60])..."' $file 2>/dev/null | head -3
        echo ""
    done
fi

echo "Quick test completed. Check detailed results in: $OUTPUT_DIR" 