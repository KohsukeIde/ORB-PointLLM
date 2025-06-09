#!/bin/bash

# Original In-Distribution Prompts Test Script
# 元のPointLLMで使われていたinstruction-tuningプロンプトをマルチクラウドで試すテスト

# CUDAモジュールの読み込み（GPU使用可能にする）
module load cuda/12.6 2>/dev/null || echo "CUDA module not available, using CPU"

DATA_PATH="/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat"
MODEL_NAME="RunsenXu/PointLLM_7B_v1.2"
OUTPUT_DIR="/groups/gag51404/ide/ORB-PointLLM/pointllm/eval/multi_pc/original_prompts_results"

echo "=== Original In-Distribution Prompts Test (ORB-PointLLM) ==="
echo "Testing original PointLLM prompts: 'What is this?', 'This is an object of'"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_NAME" 
echo "CUDA: $(which nvcc 2>/dev/null && echo 'Available' || echo 'Not available - using CPU')"

# データが存在するか確認
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

echo "Data file found: $(ls -lh $DATA_PATH | awk '{print $5}')"

# 1. 単一点群 - Original prompt "What is this?" (prompt_index=0)
echo ""
echo "1. Single Cloud - Original 'What is this?' prompt..."
python ../core/eval_multi_cloud.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --task_type object_identification \
    --num_clouds 1 \
    --num_samples 20 \
    --subset_nums 100 \
    --use_color \
    --pointnum 8192 \
    --output_dir "$OUTPUT_DIR/single_cloud" \
    --use_original_prompts \
    --prompt_index 0 \
    --force_regenerate

# 2. 単一点群 - Original prompt "This is an object of" (prompt_index=1)  
echo ""
echo "2. Single Cloud - Original 'This is an object of' prompt..."
python ../core/eval_multi_cloud.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --task_type object_identification \
    --num_clouds 1 \
    --num_samples 20 \
    --subset_nums 100 \
    --use_color \
    --pointnum 8192 \
    --output_dir "$OUTPUT_DIR/single_cloud" \
    --use_original_prompts \
    --prompt_index 1 \
    --force_regenerate

# 3. マルチクラウド - Shape Matching with original style prompts
echo ""
echo "3. Multi-Cloud Shape Matching - Original style prompts..."
python ../core/eval_multi_cloud.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --task_type shape_matching \
    --num_clouds 2 \
    --num_samples 15 \
    --subset_nums 100 \
    --use_color \
    --pointnum 8192 \
    --output_dir "$OUTPUT_DIR/multi_cloud" \
    --force_enable_multi_cloud \
    --max_point_clouds 8 \
    --use_original_prompts \
    --prompt_index 2 \
    --force_regenerate

# 4. マルチクラウド - Object Identification with original style prompts
echo ""
echo "4. Multi-Cloud Object Identification - Original style prompts..."
python ../core/eval_multi_cloud.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --task_type object_identification \
    --num_clouds 2 \
    --num_samples 15 \
    --subset_nums 100 \
    --use_color \
    --pointnum 8192 \
    --output_dir "$OUTPUT_DIR/multi_cloud" \
    --force_enable_multi_cloud \
    --max_point_clouds 8 \
    --use_original_prompts \
    --prompt_index 0 \
    --force_regenerate

# 5. 比較用 - カスタムプロンプト（既存のMULTI_CLOUD_PROMPTS）
echo ""
echo "5. Comparison - Custom multi-cloud prompts (for reference)..."
python ../core/eval_multi_cloud.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --task_type shape_matching \
    --num_clouds 2 \
    --num_samples 10 \
    --subset_nums 100 \
    --use_color \
    --pointnum 8192 \
    --output_dir "$OUTPUT_DIR/comparison" \
    --force_enable_multi_cloud \
    --max_point_clouds 8 \
    --prompt_index 0 \
    --force_regenerate

echo ""
echo "=== Original Prompts Test Complete ==="
echo "Check results in: $OUTPUT_DIR"

# 結果の表示
if [ -d "$OUTPUT_DIR" ]; then
    echo ""
    echo "Generated files:"
    find "$OUTPUT_DIR" -name "*.json" -exec echo "  {}" \; 
    
    echo ""
    echo "Prompt comparison summary:"
    for file in $(find "$OUTPUT_DIR" -name "*.json"); do
        if [ -f "$file" ]; then
            echo "File: $(basename $file)"
            echo "  Task: $(jq -r '.task_type' $file 2>/dev/null || echo 'N/A')"
            echo "  Prompt template: $(jq -r '.prompt_template' $file 2>/dev/null || echo 'N/A')"
            echo "  Multi-cloud enabled: $(jq -r '.model_config.enable_multi_cloud' $file 2>/dev/null || echo 'N/A')"
            echo ""
        fi
    done
    
    echo ""
    echo "=== Sample Results Preview ==="
    
    # Single cloud original prompts results
    single_files=$(find "$OUTPUT_DIR/single_cloud" -name "*original*.json" | head -1)
    if [ -n "$single_files" ]; then
        echo "Single Cloud Original Prompt Results:"
        echo "$(jq -r '.results[0:2][] | "Sample \(.sample_id): \(.ground_truth.label_names) -> \(.model_output[0:80])..."' $single_files 2>/dev/null)"
        echo ""
    fi
    
    # Multi-cloud original prompts results  
    multi_files=$(find "$OUTPUT_DIR/multi_cloud" -name "*original*.json" | head -1)
    if [ -n "$multi_files" ]; then
        echo "Multi-Cloud Original Prompt Results:"
        echo "$(jq -r '.results[0:2][] | "Sample \(.sample_id): \(.ground_truth.label_names | join(" vs ")) -> \(.model_output[0:80])..."' $multi_files 2>/dev/null)"
        echo ""
    fi
fi 