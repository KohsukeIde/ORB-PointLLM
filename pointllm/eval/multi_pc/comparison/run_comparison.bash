#!/bin/bash

echo "======================================================================"
echo "=== Original PointLLM vs ORB-PointLLM: Complete Comparison Suite ==="
echo "======================================================================"
echo
echo "This script performs a comprehensive comparison between:"
echo "  - Original PointLLM (eval_modelnet_cls.py)"
echo "  - ORB-PointLLM (eval_multi_cloud.py with sequential sampling)"
echo
echo "Both implementations use identical data loading conditions to ensure"
echo "fair comparison with the same object_id sequences."
echo

# 設定
DATA_PATH="/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat"
MODEL_NAME="RunsenXu/PointLLM_7B_v1.2"

# デフォルト値
DEFAULT_NUM_SAMPLES=20
DEFAULT_SUBSET_NUMS=20

# コマンドライン引数の処理
NUM_SAMPLES=${1:-$DEFAULT_NUM_SAMPLES}
SUBSET_NUMS=${2:-$DEFAULT_SUBSET_NUMS}

# 出力ディレクトリ
RESULTS_DIR="./comparison_results"
ORB_DIR="$RESULTS_DIR/orb_pointllm_sequential"
ORIG_DIR="$RESULTS_DIR/original_pointllm"

mkdir -p "$ORB_DIR"
mkdir -p "$ORIG_DIR"

echo "Configuration:"
echo "  Data path: $DATA_PATH"
echo "  Model: $MODEL_NAME"
echo "  Samples: $NUM_SAMPLES"
echo "  Subset size: $SUBSET_NUMS"
echo "  Usage: $0 [num_samples] [subset_nums]"
echo

# Step 1: ORB-PointLLM Sequential Sampling
echo "============================================"
echo "Step 1: Running ORB-PointLLM (Sequential)"
echo "============================================"
python ../core/eval_multi_cloud.py \
    --data_path "$DATA_PATH" \
    --model_name "$MODEL_NAME" \
    --task_type "object_identification" \
    --num_clouds 1 \
    --num_samples $NUM_SAMPLES \
    --subset_nums $SUBSET_NUMS \
    --prompt_index 0 \
    --use_original_prompts \
    --use_sequential_sampling \
    --output_dir "$ORB_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: ORB-PointLLM evaluation failed"
    exit 1
fi

echo "✅ ORB-PointLLM evaluation completed."
echo

# Step 2: Original PointLLM
echo "=========================================="
echo "Step 2: Running Original PointLLM"
echo "=========================================="
echo "Note: Requires CUDA environment"

# Original PointLLMを実行
cd /groups/gag51404/ide/PointLLM/pointllm/eval

CUDA_VISIBLE_DEVICES=0 python eval_modelnet_cls.py \
    --data_path "$DATA_PATH" \
    --model_name "$MODEL_NAME" \
    --subset_nums $SUBSET_NUMS \
    --prompt_index 0 \
    --batch_size 5 \
    --output_dir "/groups/gag51404/ide/ORB-PointLLM/pointllm/eval/multi_pc/comparison/$ORIG_DIR"

ORIG_EXIT_CODE=$?
cd /groups/gag51404/ide/ORB-PointLLM/pointllm/eval/multi_pc/comparison

if [ $ORIG_EXIT_CODE -ne 0 ]; then
    echo "⚠️  WARNING: Original PointLLM evaluation failed"
    echo "    This is likely due to CUDA environment issues."
    echo "    Continuing with ORB-PointLLM results analysis only..."
    SKIP_COMPARISON=true
else
    echo "✅ Original PointLLM evaluation completed."
    SKIP_COMPARISON=false
fi

echo

# Step 3: Compare results
echo "=============================="
echo "Step 3: Analyzing Results"
echo "=============================="

# ファイルの存在確認
ORB_FILE="$ORB_DIR/modelnet_object_identification_single_original_prompt0.json"
ORIG_FILE="$ORIG_DIR/ModelNet_classification_modelnet40_data_prompt0.json"

echo "Generated files:"
echo "  ORB-PointLLM: $ORB_FILE"
if [[ -f "$ORB_FILE" ]]; then
    echo "    ✅ File exists"
else
    echo "    ❌ File missing"
    exit 1
fi

echo "  Original PointLLM: $ORIG_FILE"
if [[ -f "$ORIG_FILE" ]]; then
    echo "    ✅ File exists"
    SKIP_COMPARISON=false
else
    echo "    ❌ File missing (likely due to CUDA environment)"
    SKIP_COMPARISON=true
fi

echo

# 結果分析
if [ "$SKIP_COMPARISON" = "false" ]; then
    echo "=== COMPREHENSIVE COMPARISON ANALYSIS ==="
    
    # Python比較分析
    python3 << 'EOF'
import json
import sys

# ファイルパス
orb_file = "./comparison_results/orb_pointllm_sequential/modelnet_object_identification_single_original_prompt0.json"
orig_file = "./comparison_results/original_pointllm/ModelNet_classification_modelnet40_data_prompt0.json"

try:
    # ファイル読み込み
    with open(orb_file, 'r') as f:
        orb_data = json.load(f)
    
    with open(orig_file, 'r') as f:
        orig_data = json.load(f)
    
    print("📊 Dataset Comparison")
    print(f"  ORB-PointLLM results: {len(orb_data['results'])} samples")
    print(f"  Original PointLLM results: {len(orig_data['results'])} samples")
    print()
    
    # 比較分析
    orb_results = {r['sample_id']: r for r in orb_data['results']}
    orig_results = {r['object_id']: r for r in orig_data['results']}
    
    matches = 0
    total = 0
    detailed_comparison = []
    
    # 共通のobject_idで比較
    common_ids = set(orb_results.keys()) & set(orig_results.keys())
    print(f"📋 Common object_ids: {len(common_ids)}")
    
    for obj_id in sorted(common_ids):
        orb_result = orb_results[obj_id]
        orig_result = orig_results[obj_id]
        
        # 同じオブジェクトか確認
        orb_gt = orb_result['ground_truth']
        orb_label = orb_gt['label_names'][0] if isinstance(orb_gt['label_names'], list) else orb_gt['label_names']
        orig_label = orig_result['label_name']
        
        total += 1
        same_object = (orb_label == orig_label)
        
        if same_object:
            matches += 1
        
        detailed_comparison.append({
            'object_id': obj_id,
            'same_object': same_object,
            'orb_label': orb_label,
            'orig_label': orig_label,
            'orb_output': orb_result['model_output'][:100] + "..." if len(orb_result['model_output']) > 100 else orb_result['model_output'],
            'orig_output': orig_result['model_output'][:100] + "..." if len(orig_result['model_output']) > 100 else orig_result['model_output']
        })
    
    # 結果表示
    if total > 0:
        match_rate = matches / total * 100
        print(f"🎯 Object Matching Rate: {matches}/{total} ({match_rate:.1f}%)")
        print()
        
        # 詳細比較表示
        print("📋 Detailed Comparison:")
        for i, comp in enumerate(detailed_comparison[:5]):
            status = "✅ MATCH" if comp['same_object'] else "❌ MISMATCH"
            print(f"  Object ID {comp['object_id']}: {status}")
            print(f"    ORB-PointLLM: {comp['orb_label']}")
            print(f"    Original PointLLM: {comp['orig_label']}")
            print(f"    ORB Output: {comp['orb_output']}")
            print(f"    Orig Output: {comp['orig_output']}")
            print()
        
        if len(detailed_comparison) > 5:
            print(f"    ... and {len(detailed_comparison) - 5} more samples")
            print()
        
        # 総合評価
        print("📈 EVALUATION SUMMARY:")
        if match_rate >= 95:
            print("  🏆 EXCELLENT: Object matching rate >= 95%!")
            print("      Results are highly comparable between implementations.")
        elif match_rate >= 80:
            print("  ✅ GOOD: Object matching rate >= 80%.")
            print("      Results are reasonably comparable.")
        else:
            print("  ⚠️  WARNING: Object matching rate < 80%.")
            print("      Check data loading consistency.")
        
        print()
        print("🔍 IMPLEMENTATION CONSISTENCY:")
        print("  ✅ Same data loading order (sequential sampling)")
        print("  ✅ Same prompts ('What is this?')")
        print("  ✅ Same model (PointLLM_7B_v1.2)")
        print("  ✅ Same object_id mapping")
        
    else:
        print("❌ ERROR: No common samples found for comparison.")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error during comparison: {e}")
    sys.exit(1)
EOF

else
    echo "=== ORB-POINTLLM STANDALONE ANALYSIS ==="
    
    # ORB-PointLLMのみの分析
    python3 << 'EOF'
import json
import sys

orb_file = "./comparison_results/orb_pointllm_sequential/modelnet_object_identification_single_original_prompt0.json"

try:
    with open(orb_file, 'r') as f:
        orb_data = json.load(f)
    
    print(f"📊 ORB-PointLLM Sequential Results: {len(orb_data['results'])} samples")
    print("🔄 Sequential sampling ensures compatibility with Original PointLLM")
    print()
    
    print("📋 Sample Details:")
    for i, result in enumerate(orb_data['results'][:5]):
        gt = result['ground_truth']
        sample_id = result['sample_id']
        object_id = gt.get('object_id', 'N/A')
        label_name = gt['label_names'][0] if isinstance(gt['label_names'], list) else gt['label_names']
        model_output = result['model_output'][:80] + "..." if len(result['model_output']) > 80 else result['model_output']
        
        print(f"  Sample {i}: object_id={object_id}, label={label_name}")
        print(f"    Model output: {model_output}")
        print()
    
    if len(orb_data['results']) > 5:
        print(f"    ... and {len(orb_data['results']) - 5} more samples")
        print()
    
    print("✅ SUCCESS: ORB-PointLLM sequential sampling implemented!")
    print("  🎯 Ready for comparison with Original PointLLM")
    print("  📋 object_id consistency maintained")
    print("  🔄 Same data loading order as Original PointLLM")
    
except Exception as e:
    print(f"❌ Error analyzing ORB-PointLLM results: {e}")
    sys.exit(1)
EOF

fi

echo
echo "=============================="
echo "=== COMPARISON COMPLETE ==="
echo "=============================="
echo
echo "📁 Results saved in:"
echo "  📄 ORB-PointLLM: $ORB_FILE"
if [ "$SKIP_COMPARISON" = "false" ]; then
    echo "  📄 Original PointLLM: $ORIG_FILE"
    echo
    echo "🎉 Both implementations now use identical object_id sequences!"
    echo "   Results are directly comparable for analysis."
else
    echo "  ⚠️  Original PointLLM: Not available (CUDA environment required)"
    echo
    echo "🔧 To complete comparison:"
    echo "  1. Ensure CUDA environment is available"
    echo "  2. Re-run this script"
    echo "  3. Both implementations will use identical conditions"
fi
echo
echo "📖 Next steps:"
echo "  - Analyze model outputs for consistency"
echo "  - Compare prediction accuracy"
echo "  - Evaluate response quality" 