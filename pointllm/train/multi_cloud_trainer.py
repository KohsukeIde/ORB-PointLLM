import logging
import os
from typing import Dict, Optional, List
import torch
import torch.nn as nn
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import json

logger = logging.getLogger(__name__)

class MultiCloudPointLLMTrainer(Trainer):
    """マルチクラウド対応のカスタムトレーナー"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_cloud_loss_weight = getattr(self.args, 'multi_cloud_loss_weight', 0.1)
        self.debug_multi_cloud = getattr(self.args, 'debug_multi_cloud', False)
        self.log_embedding_stats = getattr(self.args, 'log_embedding_stats', False)
        
        # 統計情報の追跡
        self.multi_cloud_stats = {
            'total_batches': 0,
            'multi_cloud_batches': 0,
            'single_cloud_batches': 0,
            'avg_clouds_per_batch': 0.0,
            'embedding_stats': []
        }
        
        logger.info(f"MultiCloudPointLLMTrainer initialized with loss_weight={self.multi_cloud_loss_weight}")

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        マルチクラウド対応の損失計算
        """
        if self.debug_multi_cloud:
            self._log_batch_info(inputs)
        
        # 基本的な損失計算
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
            
        outputs = model(**inputs)
        
        if labels is not None:
            # 言語モデルの損失
            if self.model.config.problem_type is None:
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs[0]
                
                loss_fct = nn.CrossEntropyLoss()
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            # マルチクラウド用の追加損失
            if self._is_multi_cloud_batch(inputs):
                multi_cloud_loss = self._compute_multi_cloud_loss(model, inputs, outputs)
                total_loss = loss + self.multi_cloud_loss_weight * multi_cloud_loss
                
                if self.debug_multi_cloud:
                    logger.info(f"Base loss: {loss.item():.4f}, Multi-cloud loss: {multi_cloud_loss.item():.4f}, Total: {total_loss.item():.4f}")
            else:
                total_loss = loss
                
        else:
            total_loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        # 統計更新
        self._update_stats(inputs)
        
        return (total_loss, outputs) if return_outputs else total_loss

    def _is_multi_cloud_batch(self, inputs):
        """バッチがマルチクラウドデータを含むかチェック"""
        return inputs.get('has_multi_cloud', False) or 'point_clouds_list' in inputs

    def _compute_multi_cloud_loss(self, model, inputs, outputs):
        """マルチクラウド固有の損失を計算"""
        multi_cloud_loss = torch.tensor(0.0, device=model.device)
        
        # Segment Embeddingの正則化損失
        if hasattr(model.model, 'point_segment_embedding'):
            segment_weights = model.model.point_segment_embedding.segment_embeddings.weight
            # 異なるセグメント間の距離を最大化
            pairwise_distances = torch.pdist(segment_weights, p=2)
            segment_loss = -torch.mean(pairwise_distances)  # 負の値で距離を最大化
            multi_cloud_loss += 0.1 * segment_loss
        
        # ローカル位置埋め込みの正則化
        if hasattr(model.model, 'point_local_position_embedding'):
            pos_weights = model.model.point_local_position_embedding.local_position_embeddings.weight
            # 位置埋め込みの滑らかさを保つ
            pos_diff = pos_weights[1:] - pos_weights[:-1]
            smoothness_loss = torch.mean(torch.norm(pos_diff, dim=1))
            multi_cloud_loss += 0.05 * smoothness_loss
        
        return multi_cloud_loss

    def _log_batch_info(self, inputs):
        """バッチ情報のログ出力"""
        batch_size = inputs['input_ids'].shape[0]
        has_multi_cloud = self._is_multi_cloud_batch(inputs)
        
        if has_multi_cloud and 'point_clouds_list' in inputs:
            pc_lists = inputs['point_clouds_list']
            total_clouds = sum(len([pc for pc in pc_list if pc is not None]) for pc_list in pc_lists)
            avg_clouds = total_clouds / batch_size if batch_size > 0 else 0
            logger.info(f"Multi-cloud batch: size={batch_size}, total_clouds={total_clouds}, avg_clouds={avg_clouds:.2f}")
        else:
            logger.info(f"Single-cloud batch: size={batch_size}")

    def _update_stats(self, inputs):
        """統計情報の更新"""
        self.multi_cloud_stats['total_batches'] += 1
        
        if self._is_multi_cloud_batch(inputs):
            self.multi_cloud_stats['multi_cloud_batches'] += 1
            if 'point_clouds_list' in inputs:
                total_clouds = sum(
                    len([pc for pc in pc_list if pc is not None]) 
                    for pc_list in inputs['point_clouds_list']
                )
                self.multi_cloud_stats['avg_clouds_per_batch'] = (
                    (self.multi_cloud_stats['avg_clouds_per_batch'] * (self.multi_cloud_stats['multi_cloud_batches'] - 1) + total_clouds) /
                    self.multi_cloud_stats['multi_cloud_batches']
                )
        else:
            self.multi_cloud_stats['single_cloud_batches'] += 1

    def log(self, logs: Dict[str, float]) -> None:
        """ログに統計情報を追加"""
        super().log(logs)
        
        # マルチクラウド統計の追加
        if self.multi_cloud_stats['total_batches'] > 0:
            logs.update({
                'multi_cloud_ratio': self.multi_cloud_stats['multi_cloud_batches'] / self.multi_cloud_stats['total_batches'],
                'avg_clouds_per_batch': self.multi_cloud_stats['avg_clouds_per_batch']
            })
        
        # 埋め込み統計の記録
        if self.log_embedding_stats and hasattr(self.model.model, 'point_segment_embedding'):
            self._log_embedding_statistics()

    def _log_embedding_statistics(self):
        """埋め込み統計のログ出力"""
        model = self.model.model
        
        if hasattr(model, 'point_segment_embedding'):
            segment_weights = model.point_segment_embedding.segment_embeddings.weight.data
            segment_stats = {
                'segment_mean': float(segment_weights.mean()),
                'segment_std': float(segment_weights.std()),
                'segment_norm': float(torch.norm(segment_weights, dim=1).mean())
            }
            
            logger.info(f"Segment embedding stats: {segment_stats}")
        
        if hasattr(model, 'point_local_position_embedding'):
            pos_weights = model.point_local_position_embedding.local_position_embeddings.weight.data
            pos_stats = {
                'position_mean': float(pos_weights.mean()),
                'position_std': float(pos_weights.std()),
                'position_norm': float(torch.norm(pos_weights, dim=1).mean())
            }
            
            logger.info(f"Position embedding stats: {pos_stats}")

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """モデル保存時に統計情報も保存"""
        super().save_model(output_dir, _internal_call)
        
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # 統計情報の保存
        stats_file = os.path.join(output_dir, "multi_cloud_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(self.multi_cloud_stats, f, indent=2)
        
        logger.info(f"Multi-cloud statistics saved to {stats_file}")

    def _save_checkpoint(self, model, trial, metrics=None):
        """チェックポイント保存時の処理"""
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        
        # 標準のチェックポイント保存
        super()._save_checkpoint(model, trial, metrics)
        
        # マルチクラウド固有の情報保存
        multi_cloud_info = {
            'global_step': self.state.global_step,
            'multi_cloud_stats': self.multi_cloud_stats,
            'multi_cloud_loss_weight': self.multi_cloud_loss_weight,
            'enable_multi_cloud': getattr(self.model.config, 'enable_multi_cloud', False),
            'max_point_clouds': getattr(self.model.config, 'max_point_clouds', 8)
        }
        
        info_file = os.path.join(output_dir, "multi_cloud_info.json")
        with open(info_file, 'w') as f:
            json.dump(multi_cloud_info, f, indent=2)

class MultiCloudDebugTrainer(MultiCloudPointLLMTrainer):
    """デバッグ用の拡張トレーナー"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_step_interval = getattr(self.args, 'debug_step_interval', 50)
        self.save_debug_samples = getattr(self.args, 'save_debug_samples', False)
        
    def training_step(self, model, inputs):
        """デバッグ情報付きのトレーニングステップ"""
        if self.state.global_step % self.debug_step_interval == 0:
            self._detailed_debug_log(model, inputs)
        
        return super().training_step(model, inputs)
    
    def _detailed_debug_log(self, model, inputs):
        """詳細なデバッグ情報の出力"""
        logger.info(f"=== Debug Step {self.state.global_step} ===")
        
        # 入力形状の確認
        logger.info(f"Input IDs shape: {inputs['input_ids'].shape}")
        if 'point_clouds_list' in inputs:
            for i, pc_list in enumerate(inputs['point_clouds_list']):
                non_none_pcs = [pc for pc in pc_list if pc is not None]
                logger.info(f"Sample {i}: {len(non_none_pcs)} point clouds")
                for j, pc in enumerate(non_none_pcs):
                    logger.info(f"  Cloud {j}: {pc.shape} {pc.dtype}")
        
        # モデル状態の確認
        if hasattr(model.model, 'point_segment_embedding'):
            segment_grad = model.model.point_segment_embedding.segment_embeddings.weight.grad
            if segment_grad is not None:
                logger.info(f"Segment embedding grad norm: {torch.norm(segment_grad).item():.6f}")
        
        if hasattr(model.model, 'point_local_position_embedding'):
            pos_grad = model.model.point_local_position_embedding.local_position_embeddings.weight.grad
            if pos_grad is not None:
                logger.info(f"Position embedding grad norm: {torch.norm(pos_grad).item():.6f}")
        
        logger.info("=" * 30)

def create_multi_cloud_trainer(model, tokenizer, args, **data_module):
    """マルチクラウドトレーナーのファクトリ関数"""
    if getattr(args, 'debug_multi_cloud', False):
        trainer_class = MultiCloudDebugTrainer
        logger.info("Using MultiCloudDebugTrainer")
    else:
        trainer_class = MultiCloudPointLLMTrainer
        logger.info("Using MultiCloudPointLLMTrainer")
    
    trainer = trainer_class(
        model=model,
        tokenizer=tokenizer,
        args=args,
        **data_module
    )
    
    return trainer 