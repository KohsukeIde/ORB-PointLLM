# Basic Tests

基本的な動作確認テストのディレクトリです。ORB-PointLLMのマルチクラウド機能が正常に動作するかを確認します。

## テストスクリプト

### `quick_test.bash` ⭐️ **推奨**
**基本動作の軽量確認**
- 実行時間: 5-10分
- シングル・マルチクラウド両方をテスト
- 10サンプル程度の小規模テスト

```bash
./basic_tests/quick_test.bash
```

### `gpu_test.bash`
**GPU動作確認**
- GPU環境での動作確認
- より詳細なGPUメモリ使用量チェック
- CUDA設定の検証

```bash
./basic_tests/gpu_test.bash
```

### `multi_test.bash`
**マルチクラウド機能のテスト**
- 複数の点群数でのテスト（2, 3, 4点群）
- 各種タスクでの動作確認
- より包括的なマルチクラウド機能検証

```bash
./basic_tests/multi_test.bash
```

## 推奨実行順序

1. **quick_test.bash** - まず基本動作を確認
2. **gpu_test.bash** - GPU環境での動作確認
3. **multi_test.bash** - 詳細なマルチクラウド機能確認

## 期待される結果

正常に動作している場合：
- エラーなしで実行完了
- JSONファイルが生成される
- モデル出力が合理的（オブジェクト名など）

## トラブルシューティング

### GPU関連エラー
- `CUDA_VISIBLE_DEVICES=0`が設定されているか確認
- GPUメモリ不足の場合は`--subset_nums`を小さく

### メモリ不足
- `--batch_size 1`に設定
- `--num_samples`や`--subset_nums`を減少

### データファイル不存在
- `/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat`を確認 