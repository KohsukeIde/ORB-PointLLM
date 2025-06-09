# Original In-Distribution Prompts Testing

このディレクトリには、元のPointLLMで使われていたinstruction-tuning用のin-distributionプロンプト（「What is this?」「This is an object of」など）をORB-PointLLMのマルチクラウド機能で試すためのコードとスクリプトが含まれています。

## 概要

元のPointLLMで使われていたプロンプト：
- `"What is this?"` (prompt_index=0)
- `"This is an object of"` (prompt_index=1)

これらのプロンプトをマルチクラウド対応に拡張：
- 単一点群: `"What is this?"` → そのまま使用
- マルチクラウド: `"What are these? <cloud_0> and <cloud_1>"` など

## 使用方法

### 1. クイックテスト（推奨）

```bash
# 5分程度で完了する軽量テスト
./ORB-PointLLM/pointllm/eval/multi_pc/quick_original_test.bash
```

### 2. 本格的なテスト

```bash
# 20-30分かかる包括的テスト
./ORB-PointLLM/pointllm/eval/multi_pc/test_original_prompts.bash
```

### 3. 個別実行

```bash
# 単一点群 - "What is this?"
python ORB-PointLLM/pointllm/eval/eval_multi_cloud.py \
    --model_name "RunsenXu/PointLLM_7B_v1.2" \
    --data_path "/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat" \
    --task_type object_identification \
    --num_clouds 1 \
    --num_samples 10 \
    --use_original_prompts \
    --prompt_index 0

# マルチクラウド - Original style prompts
python ORB-PointLLM/pointllm/eval/eval_multi_cloud.py \
    --model_name "RunsenXu/PointLLM_7B_v1.2" \
    --data_path "/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat" \
    --task_type object_identification \
    --num_clouds 2 \
    --num_samples 10 \
    --force_enable_multi_cloud \
    --use_original_prompts \
    --prompt_index 2
```

## 利用可能なプロンプトテンプレート

### 単一点群用 (ORIGINAL_IN_DISTRIBUTION_PROMPTS)
- `prompt_index=0`: "What is this?"
- `prompt_index=1`: "This is an object of"

### マルチクラウド用 (ORIGINAL_MULTI_CLOUD_PROMPTS)

#### object_identification
- `prompt_index=0`: "What is this? <cloud_0>"
- `prompt_index=1`: "What is this? <cloud_1>"
- `prompt_index=2`: "What are these? <cloud_0> and <cloud_1>"
- `prompt_index=3`: "Identify these objects: <cloud_0> and <cloud_1>"

#### shape_matching
- `prompt_index=0`: "What is this? <cloud_0>"
- `prompt_index=1`: "What is this? <cloud_1>"
- `prompt_index=2`: "What are these? <cloud_0> and <cloud_1>"
- `prompt_index=3`: "Identify these objects: <cloud_0> and <cloud_1>"

#### part_assembly
- `prompt_index=0`: "What are these? <cloud_0>, <cloud_1>, and <cloud_2>"
- `prompt_index=1`: "Identify these objects: <cloud_0>, <cloud_1>, <cloud_2>"

#### geometric_reasoning
- `prompt_index=0`: "What is this? <cloud_0>"
- `prompt_index=1`: "What is this? <cloud_1>"
- `prompt_index=2`: "What are these? <cloud_0> and <cloud_1>"

## 主要なオプション

### 必須オプション
- `--use_original_prompts`: Original PointLLMのin-distributionプロンプトを使用
- `--force_enable_multi_cloud`: マルチクラウド機能を強制有効化

### その他のオプション
- `--prompt_index`: プロンプトテンプレートのインデックス（0-3）
- `--num_clouds`: 使用する点群の数（1-8）
- `--num_samples`: 評価サンプル数
- `--task_type`: タスクタイプ（object_identification, shape_matching, etc.）

## 出力ファイル

結果のJSONファイルは以下の命名規則で保存されます：
```
modelnet_{task_type}_{num_clouds}clouds_original_prompt{prompt_index}.json
```

例：
- `modelnet_object_identification_single_original_prompt0.json`
- `modelnet_shape_matching_2clouds_original_prompt2.json`

## 結果の比較

生成されたJSONファイルには以下の情報が含まれます：
- `prompt_template`: 使用されたプロンプトテンプレート
- `model_config.enable_multi_cloud`: マルチクラウド機能の有効/無効
- `results[].model_output`: モデルの出力結果

通常のカスタムプロンプトとの比較により、in-distributionプロンプトの効果を評価できます。

## トラブルシューティング

### エラー: "Data file not found"
データファイルのパスを確認してください：
```bash
ls -la /groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat
```

### エラー: "CUDA out of memory"
バッチサイズを小さくするか、`--subset_nums`を使用してデータサイズを制限してください。

### マルチクラウド機能が無効
`--force_enable_multi_cloud`オプションが必要です。

## 関連ファイル

- `eval_multi_cloud.py`: メインの評価スクリプト
- `quick_original_test.bash`: クイックテスト用スクリプト
- `test_original_prompts.bash`: 包括的テスト用スクリプト 