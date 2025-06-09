# Utilities

デバッグとトラブルシューティング用のユーティリティディレクトリです。

## ユーティリティスクリプト

### `debug_dataloader.bash`
**DataLoaderサイズ確認ツール**

プログレスバーの表示がおかしい場合や、DataLoaderの動作を確認したい場合に使用します。

```bash
./utils/debug_dataloader.bash
```

**確認内容:**
- Dataset size（データセットサイズ）
- DataLoader batches（バッチ数）
- 期待されるプログレス表示
- 最初の数バッチの内容確認

**使用場面:**
- `20/10`のような奇妙なプログレスバー表示が出る場合
- DataLoaderの設定が正しいかを確認したい場合
- バッチ処理の動作を詳細に確認したい場合

## 出力例

正常な場合：
```
Dataset size: 20
DataLoader batches: 20
Expected progress: 20 batches
```

異常な場合：
- Dataset sizeとDataLoader batchesが一致しない
- 予期しないバッチサイズ

## その他のデバッグ方法

### プログレスバー問題
1. `debug_dataloader.bash`で確認
2. `subset_nums`と`num_samples`の設定を確認
3. 通常のPointLLMとORB-PointLLMで同じ設定を使用

### GPU/CUDA問題
```bash
nvidia-smi  # GPU使用状況確認
echo $CUDA_VISIBLE_DEVICES  # CUDA設定確認
```

### モデル読み込み問題
```bash
python -c "import torch; print(torch.cuda.is_available())"  # CUDA利用可能性確認
``` 