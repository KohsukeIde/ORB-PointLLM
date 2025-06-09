#!/bin/bash

# ModelNet40 Multi-Cloud Evaluation Script
# 実際のModelNet40データ(.dat)を使った評価

# データパス
DATA_PATH="/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat"
MODEL_NAME="RunsenXu/PointLLM_7B_v1.2"
OUTPUT_DIR="evaluation_results"

echo "=== ModelNet40 Multi-Cloud Evaluation ==="
echo "Data: $DATA_PATH"
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"

# 1. Shape Matching (2点群) - 同じカテゴリかどうかの比較
echo ""
echo "1. Shape Matching (2 clouds)..."
python ../core/eval_multi_cloud.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --task_type shape_matching \
    --num_clouds 2 \
    --num_samples 100 \
    --subset_nums 500 \
    --use_color \
    --pointnum 8192 \
    --output_dir "$OUTPUT_DIR/shape_matching"

# 2. Object Identification (3点群) - 複数オブジェクトの同時識別
echo ""
echo "2. Object Identification (3 clouds)..."
python ../core/eval_multi_cloud.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --task_type object_identification \
    --num_clouds 3 \
    --num_samples 50 \
    --subset_nums 300 \
    --use_color \
    --pointnum 8192 \
    --output_dir "$OUTPUT_DIR/object_identification"

# 3. Geometric Reasoning (2点群) - 幾何学的関係の推論
echo ""
echo "3. Geometric Reasoning (2 clouds)..."
python ../core/eval_multi_cloud.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --task_type geometric_reasoning \
    --num_clouds 2 \
    --num_samples 100 \
    --prompt_index 1 \
    --subset_nums 400 \
    --use_color \
    --pointnum 8192 \
    --output_dir "$OUTPUT_DIR/geometric_reasoning"

# 4. Single Cloud Baseline - 比較用の単一点群評価
echo ""
echo "4. Single Cloud Baseline..."
python ../core/eval_multi_cloud.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --task_type object_identification \
    --num_clouds 1 \
    --num_samples 200 \
    --subset_nums 500 \
    --use_color \
    --pointnum 8192 \
    --output_dir "$OUTPUT_DIR/single_baseline"

# 5. 異なるプロンプトでの評価
echo ""
echo "5. Different prompts evaluation..."
for prompt_idx in 0 1 2 3; do
    echo "  Prompt index: $prompt_idx"
    python ../core/eval_multi_cloud.py \
        --model_name "$MODEL_NAME" \
        --data_path "$DATA_PATH" \
        --task_type shape_matching \
        --num_clouds 2 \
        --num_samples 30 \
        --prompt_index $prompt_idx \
        --subset_nums 200 \
        --use_color \
        --pointnum 8192 \
        --output_dir "$OUTPUT_DIR/prompt_variations"
done

echo ""
echo "=== Evaluation Complete ==="
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Summary of files:"
find "$OUTPUT_DIR" -name "*.json" -type f | head -10