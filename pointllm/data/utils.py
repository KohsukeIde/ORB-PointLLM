from collections import OrderedDict, defaultdict

import transformers
from pointllm import conversation as conversation_lib
from dataclasses import dataclass
from typing import Optional, Dict, Sequence
import torch

import numpy as np
import os
import yaml

IGNORE_INDEX = -100

def cfg_from_yaml_file(config_file):
    """YAML設定ファイルを読み込む"""
    with open(config_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")
            return None
    
    # Namespace-like object for backward compatibility
    class ConfigDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        def __setattr__(self, name, value):
            self[name] = value
    
    return ConfigDict(config)

# * Sample Usage:
# * from utils import LRUCache
# * cache = LRUCache(capacity, max_access_count)
# if self.cache is None:
#     info_data = self.multiview_scannet[info_index]
# else:
#     info_data = self.cache.get(info_index)
#     if info_data is None or self.cache.get_access_count(info_index) >= self.cache.max_access_count:
#         # If not in cache, or accessed max_access_count times, load it and put it in cache
#         info_data = self.multiview_scannet[info_index]
#         self.cache.put(info_index, info_data)
#         self.cache.reset_access_count(info_index)

class LRUCache:
    def __init__(self, capacity, max_access_count):
        self.cache = OrderedDict()
        self.access_count = defaultdict(int)
        self.capacity = capacity
        self.max_access_count = max_access_count

    def get(self, key):
        if key not in self.cache:
            return None
        value = self.cache.pop(key)
        self.cache[key] = value  # Put key as the newest one
        self.access_count[key] += 1
        return value

    def put(self, key, value):
        if key in self.cache:  # Update the value and put it as newest
            self.cache.pop(key)
        elif len(self.cache) == self.capacity:  # If cache is full
            oldest_key = next(iter(self.cache))
            self.cache.popitem(last=False)  # Remove oldest item
            del self.access_count[oldest_key]  # Remove the corresponding access count
        self.cache[key] = value
        self.access_count[key] = 1

    def get_access_count(self, key):
        return self.access_count.get(key, 0)

    def reset_access_count(self, key):
        self.access_count[key] = 0


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2: # * can handle padded tokens
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX # * this is necessary for padded tokens

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len: # * unk tokens in the dialogue will cause this.
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_multimodal_point_cloud(
    sources: Sequence[str],
    point_backbone_config: dict,
    point_indicator: str = "<point>",
) -> Dict:
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']

    for source in sources:
        for sentence in source:
            replace_token = default_point_patch_token * point_token_len 
            if point_backbone_config['mm_use_point_start_end']:
                replace_token = point_backbone_config['default_point_start_token']+ replace_token + point_backbone_config['default_point_end_token']
            sentence["value"] = sentence["value"].replace(point_indicator, replace_token)

    return sources

def pc_norm(pc):
    """ pc: NxC, return NxC """
    xyz = pc[:, :3]
    other_feature = pc[:, 3:]

    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / m

    pc = np.concatenate((xyz, other_feature), axis=1)
    return pc

def load_objaverse_point_cloud(data_path, object_id, pointnum=8192, use_color=False):
    filename = f"{object_id}_{pointnum}.npy"
    point_cloud = np.load(os.path.join(data_path, filename))

    # * normalize
    point_cloud = pc_norm(point_cloud)

    if not use_color:
        point_cloud = point_cloud[:, :3]

    return point_cloud

@dataclass
class DataCollatorForPointTextDataset(object):
    """Collate examples for mixed dataset with text and point cloud data."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'point_clouds' in instances[0]:
            point_clouds = [instance['point_clouds'] for instance in instances]
            if all(x is not None and x.shape == point_clouds[0].shape for x in point_clouds): # * point_clouds have different shapes
                batch['point_clouds'] = torch.stack(point_clouds)
            else:
                batch['point_clouds'] = point_clouds # * return as lists

        return batch

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def pc_normalize(pc):
    """
    pc: Nx3 array
    This functions normalizes a point cloud to fit within a unit sphere.
    It first calculates the centroid of the point cloud and then subtracts
    it from all points before scaling all points to fit within a unit sphere.
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def preprocess_multimodal_multi_cloud(
    sources: Sequence[str],
    point_backbone_config: dict,
    multi_cloud_indicators: list,
) -> Dict:
    """複数点群用の前処理関数"""
    point_token_len = point_backbone_config['point_token_len']
    cloud_token_ids = point_backbone_config.get('cloud_token_ids', {})
    
    for source in sources:
        for sentence in source:
            # 各クラウドインジケーターを対応するトークンに置換
            for i, indicator in enumerate(multi_cloud_indicators):
                if indicator in sentence["value"]:
                    # クラウドトークンの取得
                    start_token = cloud_token_ids.get(f'cloud_{i}_start', f'<cloud_{i}>')
                    end_token = cloud_token_ids.get(f'cloud_{i}_end', f'</cloud_{i}>')
                    
                    # 点群トークンの作成
                    replace_token = point_backbone_config['default_point_patch_token'] * point_token_len
                    if point_backbone_config['mm_use_point_start_end']:
                        replace_token = point_backbone_config['default_point_start_token'] + replace_token + point_backbone_config['default_point_end_token']
                    
                    # 最終的な置換文字列
                    final_token = start_token + replace_token + end_token
                    sentence["value"] = sentence["value"].replace(indicator, final_token)
    
    return sources

@dataclass
class MultiCloudDataCollatorForPointTextDataset(object):
    """マルチクラウド対応のコレーター"""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # マルチクラウドとシングルクラウドの混在対応
        has_multi_cloud = any(instance.get('is_multi_cloud', False) for instance in instances)
        has_single_cloud = any('point_clouds' in instance for instance in instances)
        
        if has_multi_cloud:
            # マルチクラウドデータの処理
            point_clouds_lists = []
            for instance in instances:
                if instance.get('is_multi_cloud', False):
                    point_clouds_lists.append(instance.get('point_clouds_list', []))
                elif 'point_clouds' in instance:
                    # シングルクラウドをマルチクラウド形式に変換
                    point_clouds_lists.append([instance['point_clouds']])
                else:
                    point_clouds_lists.append([])
            
            batch['point_clouds_list'] = point_clouds_lists
            batch['has_multi_cloud'] = True
        
        if has_single_cloud and not has_multi_cloud:
            # 従来のシングルクラウド処理
            point_clouds = [instance['point_clouds'] for instance in instances if 'point_clouds' in instance]
            if point_clouds:
                if all(x is not None and x.shape == point_clouds[0].shape for x in point_clouds):
                    batch['point_clouds'] = torch.stack(point_clouds)
                else:
                    batch['point_clouds'] = point_clouds
            batch['has_multi_cloud'] = False

        return batch

# ===== 新規: マルチクラウド用のユーティリティ関数 =====

def load_multi_cloud_point_clouds(data_path, object_ids, pointnum=8192, use_color=False, normalize=True):
    """複数の点群を読み込む"""
    point_clouds = []
    for object_id in object_ids:
        if object_id is not None:
            try:
                pc = load_objaverse_point_cloud(data_path, object_id, pointnum, use_color)
                if normalize:
                    pc = pc_norm(pc)
                point_clouds.append(pc)
            except Exception as e:
                print(f"Warning: Failed to load {object_id}: {e}")
                point_clouds.append(None)
        else:
            point_clouds.append(None)
    return point_clouds

def validate_multi_cloud_batch(batch, logger=None):
    """マルチクラウドバッチの検証"""
    if 'point_clouds_list' not in batch:
        return True
    
    point_clouds_lists = batch['point_clouds_list']
    for i, pc_list in enumerate(point_clouds_lists):
        if not isinstance(pc_list, list):
            if logger:
                logger.warning(f"Sample {i}: point_clouds_list is not a list")
            return False
        
        for j, pc in enumerate(pc_list):
            if pc is not None and not isinstance(pc, torch.Tensor):
                if logger:
                    logger.warning(f"Sample {i}, Cloud {j}: not a tensor")
                return False
    
    return True

def create_multi_cloud_conversation_template():
    """マルチクラウド用の会話テンプレート作成"""
    templates = {
        'shape_matching': [
            "Compare these two 3D shapes: <cloud_0> and <cloud_1>. Are they compatible for assembly?",
            "Analyze the geometric compatibility between <cloud_0> and <cloud_1>.",
            "Determine if <cloud_0> can be properly connected to <cloud_1>."
        ],
        'part_assembly': [
            "Given these parts <cloud_0>, <cloud_1>, and <cloud_2>, what is the correct assembly order?",
            "How should these components be assembled: <cloud_0>, <cloud_1>, <cloud_2>?",
            "Describe the step-by-step assembly process for these parts: <cloud_0>, <cloud_1>, <cloud_2>."
        ],
        'geometric_reasoning': [
            "What is the spatial relationship between <cloud_0> and <cloud_1>?",
            "Describe how <cloud_0> is positioned relative to <cloud_1>.",
            "Compare the sizes and orientations of <cloud_0> and <cloud_1>."
        ]
    }
    return templates