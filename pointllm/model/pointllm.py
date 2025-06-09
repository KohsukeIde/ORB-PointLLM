#    Copyright 2023 Runsen Xu

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from .utils import *
from pointllm.utils import *

from contextlib import nullcontext
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

import os

# * add logger
import logging
logger = logging.getLogger(__name__)

# ===== 新規: Segment Embedding と ローカル位置ID の実装 =====

class PointCloudSegmentEmbedding(nn.Module):
    """
    複数点群用のSegment Embedding
    各点群に異なるセグメントIDを付与
    """
    def __init__(self, hidden_size: int, max_segments: int = 8):
        super().__init__()
        self.max_segments = max_segments
        self.segment_embeddings = nn.Embedding(max_segments, hidden_size)
        
        # 初期化
        nn.init.normal_(self.segment_embeddings.weight, mean=0.0, std=0.02)
        
    def forward(self, segment_ids: torch.Tensor):
        """
        Args:
            segment_ids: [batch_size, seq_len] - 各トークンのセグメントID
        Returns:
            segment_embeds: [batch_size, seq_len, hidden_size]
        """
        return self.segment_embeddings(segment_ids)

class PointCloudLocalPositionEmbedding(nn.Module):
    """
    複数点群用のローカル位置埋め込み
    各点群内で位置IDをリセット
    """
    def __init__(self, hidden_size: int, max_position: int = 1024):
        super().__init__()
        self.max_position = max_position
        self.local_position_embeddings = nn.Embedding(max_position, hidden_size)
        
        # 初期化
        nn.init.normal_(self.local_position_embeddings.weight, mean=0.0, std=0.02)
        
    def forward(self, local_position_ids: torch.Tensor):
        """
        Args:
            local_position_ids: [batch_size, seq_len] - 各トークンのローカル位置ID
        Returns:
            local_pos_embeds: [batch_size, seq_len, hidden_size]
        """
        return self.local_position_embeddings(local_position_ids)

class MultiPointCloudEmbeddingManager:
    """
    複数点群の埋め込み管理クラス
    Segment IDとローカル位置IDの生成・管理
    """
    
    @staticmethod
    def create_segment_ids(point_clouds_info: List[dict], point_token_len: int, device: torch.device):
        """
        セグメントIDの生成
        
        Args:
            point_clouds_info: 点群情報のリスト [{"num_tokens": int, "cloud_idx": int}, ...]
            point_token_len: 各点群のトークン数
            device: デバイス
            
        Returns:
            segment_ids: [total_tokens] - セグメントIDの配列
        """
        segment_ids = []
        
        for cloud_info in point_clouds_info:
            cloud_idx = cloud_info.get('cloud_idx', 0)
            num_tokens = cloud_info.get('num_tokens', point_token_len)
            
            # 各点群のセグメントIDを生成
            cloud_segment_ids = torch.full((num_tokens,), cloud_idx, dtype=torch.long, device=device)
            segment_ids.append(cloud_segment_ids)
        
        return torch.cat(segment_ids, dim=0)
    
    @staticmethod
    def create_local_position_ids(point_clouds_info: List[dict], point_token_len: int, device: torch.device):
        """
        ローカル位置IDの生成
        
        Args:
            point_clouds_info: 点群情報のリスト
            point_token_len: 各点群のトークン数
            device: デバイス
            
        Returns:
            local_position_ids: [total_tokens] - ローカル位置IDの配列
        """
        local_position_ids = []
        
        for cloud_info in point_clouds_info:
            num_tokens = cloud_info.get('num_tokens', point_token_len)
            
            # 各点群内でローカル位置IDを0からリセット
            cloud_local_ids = torch.arange(num_tokens, dtype=torch.long, device=device)
            local_position_ids.append(cloud_local_ids)
        
        return torch.cat(local_position_ids, dim=0)

# ===== PointLLMConfig クラスの拡張 =====

class PointLLMConfig(LlamaConfig):
    """PointLLMConfigにマルチクラウド設定を追加"""
    model_type = "pointllm"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 既存のPointLLM設定
        self.point_backbone = kwargs.get('point_backbone', 'PointBERT')
        self.point_backbone_config_name = kwargs.get('point_backbone_config_name', 'PointTransformer_8192point_2layer')
        self.use_color = kwargs.get('use_color', True)
        self.mm_use_point_start_end = kwargs.get('mm_use_point_start_end', False)
        
        # 新規: マルチクラウド設定
        self.enable_multi_cloud = kwargs.get('enable_multi_cloud', False)
        self.max_point_clouds = kwargs.get('max_point_clouds', 8)
        self.use_segment_embedding = kwargs.get('use_segment_embedding', True)
        self.use_local_position_embedding = kwargs.get('use_local_position_embedding', True)
        self.cross_attention_layers = kwargs.get('cross_attention_layers', 0)
        
        # 特殊トークンの設定
        self.DEFAULT_POINT_PATCH_TOKEN = "<point_patch>"
        self.DEFAULT_POINT_START_TOKEN = "<point_start>"
        self.DEFAULT_POINT_END_TOKEN = "<point_end>"

class PointLLMLlamaModel(LlamaModel):
    config_class = PointLLMConfig 

    def __init__(self, config: LlamaConfig):
        super(PointLLMLlamaModel, self).__init__(config)

        self.point_backbone_type = config.point_backbone
        logger.info(f"Using {self.point_backbone_type}.")

        if self.point_backbone_type == "PointBERT":
            from pointllm.model import PointTransformer
            # address of config file, in the same dir of this file
            point_bert_config_name = getattr(config, "point_backbone_config_name", "PointTransformer_8192point_2layer")
            point_bert_config_addr = os.path.join(os.path.dirname(__file__), "pointbert", f"{point_bert_config_name}.yaml")
            print(f"Loading PointBERT config from {point_bert_config_addr}.")
            point_bert_config = cfg_from_yaml_file(point_bert_config_addr)
            if getattr(config, "use_color", False):
                point_bert_config.model.point_dims = 6
            use_max_pool = getattr(point_bert_config.model, "use_max_pool", False)
            
            self.point_backbone = PointTransformer(point_bert_config.model, use_max_pool=use_max_pool)
            logger.info(f"Using {self.point_backbone.point_dims} dim of points.")

            self.point_backbone_config = {
                "point_cloud_dim": point_bert_config.model.point_dims,
                "backbone_output_dim": point_bert_config.model.trans_dim if not use_max_pool else point_bert_config.model.trans_dim * 2,
                "project_output_dim": self.config.hidden_size,
                "point_token_len": point_bert_config.model.num_group + 1 if not use_max_pool else 1,
                "mm_use_point_start_end": self.config.mm_use_point_start_end,
                "projection_hidden_layer": point_bert_config.model.get('projection_hidden_layer', 0),
                "use_max_pool": use_max_pool
            }
            if point_bert_config.model.get('projection_hidden_layer', 0) > 0:
                self.point_backbone_config["projection_hidden_dim"] = point_bert_config.model.projection_hidden_dim
            
        # Projector の初期化（既存コード）
        backbone_output_dim = self.point_backbone_config["backbone_output_dim"]
        logger.info(f"Point backbone output dim: {backbone_output_dim}.")
        logger.info(f"Use {self.point_backbone_config['projection_hidden_layer']} projection hidden layers.")
        if self.point_backbone_config['projection_hidden_layer'] > 0:
            projection_layers = []
            last_dim = backbone_output_dim
            for i in range(point_bert_config.model.projection_hidden_layer):
                projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["projection_hidden_dim"][i]))
                projection_layers.append(nn.GELU())
                last_dim = self.point_backbone_config["projection_hidden_dim"][i]
            projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["project_output_dim"]))
            self.point_proj = nn.Sequential(*projection_layers)
            logger.info(f"Each layer with {point_bert_config.model.projection_hidden_dim} hidden units.")
        else:
            self.point_proj = nn.Linear(backbone_output_dim, self.point_backbone_config['project_output_dim'])
        logger.info(f"Point projector output dim: {self.point_backbone_config['project_output_dim']}.")

        # ===== 新規: Segment Embedding と ローカル位置埋め込みの追加 =====
        self.enable_multi_cloud = getattr(config, 'enable_multi_cloud', False)
        self.max_point_clouds = getattr(config, 'max_point_clouds', 8)
        
        if self.enable_multi_cloud:
            logger.info("Initializing multi-cloud embeddings...")
            
            # Segment Embedding
            self.point_segment_embedding = PointCloudSegmentEmbedding(
                hidden_size=self.config.hidden_size,
                max_segments=self.max_point_clouds
            )
            
            # ローカル位置埋め込み
            max_point_tokens = self.point_backbone_config.get("point_token_len", 513)
            self.point_local_position_embedding = PointCloudLocalPositionEmbedding(
                hidden_size=self.config.hidden_size,
                max_position=max_point_tokens
            )
            
            # 埋め込み管理
            self.embedding_manager = MultiPointCloudEmbeddingManager()
            
            logger.info(f"Multi-cloud embeddings initialized: "
                       f"max_segments={self.max_point_clouds}, "
                       f"max_position={max_point_tokens}")

        self.fix_pointnet = False
        self.fix_llm = False

    def _get_token_id(self, token: str):
        """特殊トークンのIDを取得"""
        cloud_token_ids = self.point_backbone_config.get('cloud_token_ids', {})
        
        # クラウドトークンの検索
        for key, token_id in cloud_token_ids.items():
            if token in key:
                return token_id
        
        # 基本的なポイントトークンの検索
        if token == '<point>':
            return self.point_backbone_config.get('point_patch_token', 0)
        elif token == '<point_start>':
            return self.point_backbone_config.get('point_start_token', 0)
        elif token == '<point_end>':
            return self.point_backbone_config.get('point_end_token', 0)
        
        # パターンマッチング
        import re
        cloud_pattern = r'<cloud_(\d+)>'
        cloud_end_pattern = r'</cloud_(\d+)>'
        
        if re.match(cloud_pattern, token):
            cloud_idx = int(re.findall(r'\d+', token)[0])
            return cloud_token_ids.get(f'cloud_{cloud_idx}_start', 0)
        elif re.match(cloud_end_pattern, token):
            cloud_idx = int(re.findall(r'\d+', token)[0])
            return cloud_token_ids.get(f'cloud_{cloud_idx}_end', 0)
        
        return 0  # デフォルト

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        point_clouds: Optional[torch.FloatTensor] = None,
        point_clouds_list: Optional[List[torch.FloatTensor]] = None,  # 新規
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # 既存の埋め込み取得
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        point_backbone = getattr(self, 'point_backbone', None)
        point_backbone_config = getattr(self, 'point_backbone_config', None)

        # ===== 複数点群の処理 =====
        if point_backbone is not None and (input_ids.shape[1] != 1 or self.training):
            
            # 単一点群と複数点群の両方に対応
            if point_clouds_list is not None and len(point_clouds_list) > 0:
                # 複数点群の場合
                point_clouds_to_process = point_clouds_list
                is_multi_cloud = True
            elif point_clouds is not None:
                # 単一点群の場合（後方互換性）
                point_clouds_to_process = [point_clouds]
                is_multi_cloud = False
            else:
                point_clouds_to_process = None
                is_multi_cloud = False
            
            if point_clouds_to_process is not None:
                # 点群の特徴抽出
            with torch.no_grad() if self.fix_pointnet else nullcontext():
                if self.fix_pointnet:
                    self.point_backbone.eval()
                    
                    if type(point_clouds_to_process) is list:
                        point_features_list = []
                        for point_cloud in point_clouds_to_process:
                            if point_cloud is not None:
                                if point_cloud.dim() == 2:  # [N, D] -> [1, N, D]
                                    point_cloud = point_cloud.unsqueeze(0)
                                point_feature = self.point_backbone(point_cloud)
                                point_features_list.append(point_feature)
                    else:
                        point_features_list = [self.point_backbone(point_clouds_to_process)]

                # プロジェクション
                projected_features_list = []
                for features in point_features_list:
                    projected = self.point_proj(features)
                    projected_features_list.append(projected)

                # ===== Segment Embedding と ローカル位置ID の適用 =====
                if self.enable_multi_cloud and is_multi_cloud and len(projected_features_list) > 1:
                    inputs_embeds = self._apply_multi_cloud_embeddings(
                        inputs_embeds, input_ids, projected_features_list, point_backbone_config
                    )
                else:
                    # 単一点群の場合（既存の処理）
                    inputs_embeds = self._apply_single_cloud_embeddings(
                        inputs_embeds, input_ids, projected_features_list[0] if projected_features_list else None, point_backbone_config
                    )

        return super(PointLLMLlamaModel, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def _apply_multi_cloud_embeddings(self, inputs_embeds, input_ids, projected_features_list, point_backbone_config):
        """複数点群用の埋め込み適用（Segment Embedding + ローカル位置ID）"""
        
        point_token_len = point_backbone_config['point_token_len']
        new_input_embeds = []
        
        for batch_idx in range(inputs_embeds.shape[0]):
            cur_input_embeds = inputs_embeds[batch_idx]
            cur_input_ids = input_ids[batch_idx]
            
            # 点群情報の収集
            point_clouds_info = []
            point_tokens_start_positions = []
            
            for cloud_idx in range(len(projected_features_list)):
                cloud_start_token = f'<cloud_{cloud_idx}>'
                cloud_end_token = f'</cloud_{cloud_idx}>'
                
                # トークン位置の検索
                start_token_id = self._get_token_id(cloud_start_token)
                end_token_id = self._get_token_id(cloud_end_token)
                
                if start_token_id in cur_input_ids and end_token_id in cur_input_ids:
                    start_positions = (cur_input_ids == start_token_id).nonzero(as_tuple=True)[0]
                    if len(start_positions) > 0:
                        start_pos = start_positions[0].item() + 1  # トークンの次の位置から
                        point_tokens_start_positions.append(start_pos)
                        point_clouds_info.append({
                            'cloud_idx': cloud_idx,
                            'num_tokens': point_token_len,
                            'start_pos': start_pos
                        })
            
            if point_clouds_info:
                # Segment ID と ローカル位置ID の生成
                segment_ids = self.embedding_manager.create_segment_ids(
                    point_clouds_info, point_token_len, inputs_embeds.device
                )
                local_position_ids = self.embedding_manager.create_local_position_ids(
                    point_clouds_info, point_token_len, inputs_embeds.device
                )
                
                # Segment Embedding と ローカル位置埋め込みの取得
                segment_embeds = self.point_segment_embedding(segment_ids)
                local_pos_embeds = self.point_local_position_embedding(local_position_ids)
                
                # 点群特徴の統合と埋め込みの適用
                combined_features = torch.cat([feat[batch_idx] for feat in projected_features_list], dim=0)
                
                # 点群トークンの位置に特徴量と追加埋め込みを配置
                enhanced_embeds = cur_input_embeds.clone()
                feature_idx = 0
                
                for cloud_info in point_clouds_info:
                    start_pos = cloud_info['start_pos']
                    end_pos = start_pos + point_token_len
                    
                    if feature_idx + point_token_len <= combined_features.shape[0]:
                        # 点群特徴 + Segment Embedding + ローカル位置埋め込み
                        point_tokens = (
                            combined_features[feature_idx:feature_idx + point_token_len] +
                            segment_embeds[feature_idx:feature_idx + point_token_len] +
                            local_pos_embeds[feature_idx:feature_idx + point_token_len]
                        )
                        
                        enhanced_embeds[start_pos:end_pos] = point_tokens
                        feature_idx += point_token_len
                
                new_input_embeds.append(enhanced_embeds)
            else:
                new_input_embeds.append(cur_input_embeds)
        
        return torch.stack(new_input_embeds, dim=0)

    def _apply_single_cloud_embeddings(self, inputs_embeds, input_ids, point_features, point_backbone_config):
        """単一点群用の埋め込み適用（既存の処理を維持）"""
        
        if point_features is None:
            return inputs_embeds
        
        # 既存のPointLLMの処理をそのまま使用
        point_token_len = point_backbone_config['point_token_len']
        dummy_point_features = torch.zeros(
            point_token_len, 
            point_backbone_config['backbone_output_dim'], 
            device=inputs_embeds.device, 
            dtype=inputs_embeds.dtype
        )
            dummy_point_features = self.point_proj(dummy_point_features)

            new_input_embeds = []
        
        for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == point_backbone_config['point_patch_token']).sum() == 0:
                cur_input_embeds = cur_input_embeds + (0. * dummy_point_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue
                
            cur_point_features = point_features[0].to(device=cur_input_embeds.device)
            num_patches = cur_point_features.shape[0]
            
                if point_backbone_config['mm_use_point_start_end']:
                # start/end トークンを使用する場合の処理（既存コード）
                    point_start_tokens = torch.where(cur_input_ids == point_backbone_config["point_start_token"])[0]
                    for point_start_token_pos in point_start_tokens:
                        if cur_input_ids[point_start_token_pos + num_patches + 1] != point_backbone_config["point_end_token"]:
                            raise ValueError("The point end token should follow the point start token.")
                    
                    cur_new_input_embeds = torch.cat((
                        cur_input_embeds[:point_start_token_pos+1], 
                        cur_point_features, 
                        cur_input_embeds[point_start_token_pos + num_patches + 1:]
                    ), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                # パッチトークンを使用する場合の処理（既存コード）
                    masked_indices = torch.where(cur_input_ids == point_backbone_config["point_patch_token"])[0]
                    mask_index_start = masked_indices[0]
                
                cur_new_input_embeds = torch.cat((
                    cur_input_embeds[:mask_index_start], 
                    cur_point_features, 
                    cur_input_embeds[mask_index_start+num_patches:]
                ), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
        
        return torch.stack(new_input_embeds, dim=0)

    def load_point_backbone_checkpoint(self, checkpoint_path=None):
        """既存メソッド（変更なし）"""
        self.point_backbone.load_checkpoint(self.config.point_backbone_ckpt if checkpoint_path is None else checkpoint_path)

class PointLLMLlamaForCausalLM(LlamaForCausalLM):
    config_class = PointLLMConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = PointLLMLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        point_clouds: Optional[torch.FloatTensor] = None,
        point_clouds_list: Optional[List[torch.FloatTensor]] = None,  # 新規
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            point_clouds=point_clouds,
            point_clouds_list=point_clouds_list  # 新規
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "point_clouds": kwargs.get("point_clouds", None),
                "point_clouds_list": kwargs.get("point_clouds_list", None),  # 新規
            }
        )
        return model_inputs

    def initialize_tokenizer_point_backbone_config_wo_embedding(self, tokenizer):
        """既存メソッドにマルチクラウド対応を追加"""
        config = self.config
        point_backbone_config = self.get_model().point_backbone_config
        mm_use_point_start_end = point_backbone_config['mm_use_point_start_end'] = config.mm_use_point_start_end

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN
        tokenizer.add_tokens([default_point_patch_token], special_tokens=True)

        point_backbone_config['default_point_patch_token'] = default_point_patch_token
        point_backbone_config['point_patch_token'] = tokenizer.convert_tokens_to_ids([default_point_patch_token])[0]

        if mm_use_point_start_end:
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            tokenizer.add_tokens([default_point_start_token, default_point_end_token], special_tokens=True)

            point_backbone_config['default_point_start_token'] = default_point_start_token
            point_backbone_config['default_point_end_token'] = default_point_end_token
            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids([default_point_start_token])[0]
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids([default_point_end_token])[0]
        
        # ===== 新規: マルチクラウド用トークンの追加 =====
        if getattr(config, 'enable_multi_cloud', False):
            max_clouds = getattr(config, 'max_point_clouds', 8)
            multi_cloud_tokens = []
            
            for i in range(max_clouds):
                start_token = f'<cloud_{i}>'
                end_token = f'</cloud_{i}>'
                multi_cloud_tokens.extend([start_token, end_token])
            
            num_added = tokenizer.add_tokens(multi_cloud_tokens, special_tokens=True)
            
            # トークンIDをpoint_backbone_configに保存
            cloud_token_ids = {}
            for i in range(max_clouds):
                start_token = f'<cloud_{i}>'
                end_token = f'</cloud_{i}>'
                cloud_token_ids[f'cloud_{i}_start'] = tokenizer.convert_tokens_to_ids([start_token])[0]
                cloud_token_ids[f'cloud_{i}_end'] = tokenizer.convert_tokens_to_ids([end_token])[0]
            
            point_backbone_config['cloud_token_ids'] = cloud_token_ids
            point_backbone_config['multi_cloud_tokens_added'] = num_added
            
            print(f"Added {num_added} multi-cloud tokens to tokenizer")
    
    def initialize_tokenizer_point_backbone_config(self, tokenizer, device, fix_llm=True):
        """既存メソッドにマルチクラウド対応を追加"""
        config = self.config
        point_backbone_config = self.get_model().point_backbone_config
        mm_use_point_start_end = point_backbone_config['mm_use_point_start_end'] = config.mm_use_point_start_end

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN
        point_backbone_config['default_point_patch_token'] = default_point_patch_token
        tokenizer.add_tokens([default_point_patch_token], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))
        point_backbone_config['point_patch_token'] = tokenizer.convert_tokens_to_ids([default_point_patch_token])[0]

        total_new_tokens = 1  # point_patch_token

        if mm_use_point_start_end:
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            point_backbone_config['default_point_start_token'] = default_point_start_token
            point_backbone_config['default_point_end_token'] = default_point_end_token

            num_new_tokens = tokenizer.add_tokens([default_point_start_token, default_point_end_token], special_tokens=True)
            total_new_tokens += num_new_tokens
            self.resize_token_embeddings(len(tokenizer))
            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids([default_point_start_token])[0]
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids([default_point_end_token])[0]

        # ===== 新規: マルチクラウド用トークンの追加 =====
        if getattr(config, 'enable_multi_cloud', False):
            max_clouds = getattr(config, 'max_point_clouds', 8)
            multi_cloud_tokens = []
            
            for i in range(max_clouds):
                start_token = f'<cloud_{i}>'
                end_token = f'</cloud_{i}>'
                multi_cloud_tokens.extend([start_token, end_token])
            
            num_multi_cloud_tokens = tokenizer.add_tokens(multi_cloud_tokens, special_tokens=True)
            total_new_tokens += num_multi_cloud_tokens
            
            if num_multi_cloud_tokens > 0:
                self.resize_token_embeddings(len(tokenizer))
            
            # トークンIDをpoint_backbone_configに保存
            cloud_token_ids = {}
            for i in range(max_clouds):
                start_token = f'<cloud_{i}>'
                end_token = f'</cloud_{i}>'
                cloud_token_ids[f'cloud_{i}_start'] = tokenizer.convert_tokens_to_ids([start_token])[0]
                cloud_token_ids[f'cloud_{i}_end'] = tokenizer.convert_tokens_to_ids([end_token])[0]
            
            point_backbone_config['cloud_token_ids'] = cloud_token_ids
            print(f"Added {num_multi_cloud_tokens} multi-cloud tokens")

        # 新しいトークンの埋め込み初期化
        if total_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-total_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-total_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-total_new_tokens:] = input_embeddings_avg
            output_embeddings[-total_new_tokens:] = output_embeddings_avg

                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                
                if fix_llm:
                self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_output_embeddings().parameters():
                        p.requires_grad = False
                print(f"Setting output embeddings fixed and {total_new_tokens} new tokens' input embeddings trainable.")
                else:
                    self.get_model().orig_embeds_params = None
                    for p in self.get_output_embeddings().parameters():
                        p.requires_grad = True
                    print("Setting output embeddings and all input embeddings trainable.")

AutoConfig.register("pointllm", PointLLMConfig)
AutoModelForCausalLM.register(PointLLMConfig, PointLLMLlamaForCausalLM)
