# Core Evaluation System

このディレクトリには、ORB-PointLLMのメインの評価システムが含まれています。

## ファイル

### `eval_multi_cloud.py`
**メインの評価スクリプト**

マルチポイントクラウド対応の評価スクリプトです。以下の機能を提供：

- **マルチクラウド評価**: 複数の点群を同時に処理
- **シングルクラウド評価**: 従来の単一点群評価との互換性
- **タスク対応**: shape_matching, part_assembly, geometric_reasoning, object_identification
- **プロンプト対応**: カスタムプロンプトとoriginal PointLLMプロンプト

## 使用例

```bash
# シングルクラウド評価
python core/eval_multi_cloud.py \
    --model_name "RunsenXu/PointLLM_7B_v1.2" \
    --task_type object_identification \
    --num_clouds 1 \
    --num_samples 10 \
    --use_original_prompts

# マルチクラウド評価
python core/eval_multi_cloud.py \
    --model_name "RunsenXu/PointLLM_7B_v1.2" \
    --task_type shape_matching \
    --num_clouds 2 \
    --num_samples 10 \
    --force_enable_multi_cloud
```

## 主要オプション

- `--task_type`: 評価タスク
- `--num_clouds`: 点群数
- `--use_original_prompts`: Original PointLLMプロンプト使用
- `--force_enable_multi_cloud`: マルチクラウド機能強制有効化

詳細は `python core/eval_multi_cloud.py --help` を参照してください。 