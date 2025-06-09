import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import os
import json
import numpy as np
from tqdm import tqdm
import random
from typing import List, Dict, Any

import sys
import os
# ORB-PointLLMへのパスを優先して追加
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
# OriginalPointLLMもバックアップとして追加
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'PointLLM'))

from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.data import ModelNet
from transformers import AutoTokenizer

# マルチクラウド用のプロンプトテンプレート
MULTI_CLOUD_PROMPTS = {
    "shape_matching": [
        "Compare these shapes: <cloud_0> and <cloud_1>. Are they the same type of object?",
        "Look at <cloud_0> and <cloud_1>. Do these objects belong to the same category?",
        "Analyze <cloud_0> and <cloud_1>. Are these compatible or similar objects?",
        "Compare the objects <cloud_0> and <cloud_1>. What is their relationship?",
    ],
    "part_assembly": [
        "Given these parts <cloud_0>, <cloud_1>, and <cloud_2>, describe how they could be assembled.",
        "Look at these components: <cloud_0>, <cloud_1>, <cloud_2>. What is a logical assembly order?",
        "Describe the relationship between these parts: <cloud_0>, <cloud_1>, <cloud_2>.",
        "How would you combine these objects: <cloud_0>, <cloud_1>, <cloud_2>?",
    ],
    "geometric_reasoning": [
        "Compare the shapes of <cloud_0> and <cloud_1>. What are their geometric differences?",
        "Describe the geometric relationship between <cloud_0> and <cloud_1>.",
        "Analyze the structural differences between <cloud_0> and <cloud_1>.",
        "What geometric properties distinguish <cloud_0> from <cloud_1>?",
    ],
    "object_identification": [
        "Identify these objects: <cloud_0> and <cloud_1>.",
        "What are these objects: <cloud_0>, <cloud_1>?",
        "Classify each of these shapes: <cloud_0>, <cloud_1>, <cloud_2>.",
        "Name these objects: <cloud_0> and <cloud_1>.",
    ]
}

SINGLE_CLOUD_PROMPTS = [
    "What is this?",
    "Describe this 3D object.",
    "Identify this shape.",
    "What object is this?",
]

# OriginalPointLLM in-distribution prompts (instruction-tuningで使われたもの)
ORIGINAL_IN_DISTRIBUTION_PROMPTS = [
    "What is this?",
    "This is an object of "
]

# Multi-cloud対応版のoriginal prompts
ORIGINAL_MULTI_CLOUD_PROMPTS = {
    "shape_matching": [
        "What is this? <cloud_0>", 
        "What is this? <cloud_1>",
        "What are these? <cloud_0> and <cloud_1>",
        "Identify these objects: <cloud_0> and <cloud_1>",
    ],
    "object_identification": [
        "What is this? <cloud_0>",
        "What is this? <cloud_1>", 
        "What are these? <cloud_0> and <cloud_1>",
        "Identify these objects: <cloud_0> and <cloud_1>",
    ],
    "part_assembly": [
        "What are these? <cloud_0>, <cloud_1>, and <cloud_2>",
        "Identify these objects: <cloud_0>, <cloud_1>, <cloud_2>",
    ],
    "geometric_reasoning": [
        "What is this? <cloud_0>", 
        "What is this? <cloud_1>",
        "What are these? <cloud_0> and <cloud_1>",
    ]
}

class ModelNetMultiCloudDataset(Dataset):
    """ModelNet40を使ったマルチクラウド評価データセット"""
    
    def __init__(self, data_path, task_type="shape_matching", num_clouds=2, num_samples=500, 
                 pointnum=8192, use_color=True, subset_nums=-1, use_sequential_sampling=False):
        self.data_path = data_path
        self.task_type = task_type
        self.num_clouds = num_clouds
        self.pointnum = pointnum
        self.use_color = True  # PointLLMでは色情報が必須
        self.use_sequential_sampling = use_sequential_sampling
        
        print(f"Loading ModelNet40 from {data_path}")
        print(f"Task: {task_type}, Clouds per sample: {num_clouds}, Total samples: {num_samples}")
        if use_sequential_sampling:
            print(f"Sequential sampling enabled for comparison with Original PointLLM")
        
        # ModelNetデータセットを読み込み
        # PointLLMは6チャンネル（XYZ + RGB）を期待するため、use_color=Trueに統一
        if data_path and os.path.exists(data_path):
            # カスタムデータパスが指定された場合
            print(f"Load custom data from {data_path}...")
            self.modelnet = self._load_custom_modelnet_data(data_path, subset_nums)
        else:
            # デフォルト設定を使用
            self.modelnet = ModelNet(
                config_path=None,  # デフォルト設定を使用
                split="test",
                subset_nums=subset_nums,
                use_color=True  # 強制的にTrueに設定（モデルが6チャンネルを期待）
            )
        
        print(f"Loaded {len(self.modelnet)} ModelNet samples")
        
        # マルチクラウドサンプルを生成
        self.samples = self._create_multi_cloud_samples(num_samples, use_sequential_sampling)
        print(f"Created {len(self.samples)} multi-cloud samples")
    
    def _load_custom_modelnet_data(self, data_path, subset_nums):
        """カスタムModelNetデータファイルを読み込み"""
        import pickle
        
        with open(data_path, 'rb') as f:
            list_of_points, list_of_labels = pickle.load(f)
        
        if subset_nums > 0:
            import random
            random.seed(0)  # Original PointLLMと一致させる
            idxs = random.sample(range(len(list_of_labels)), min(subset_nums, len(list_of_labels)))
            list_of_points = [list_of_points[idx] for idx in idxs]
            list_of_labels = [list_of_labels[idx] for idx in idxs]
            print(f"🎯 Selected indices with seed 0: {idxs[:10] if len(idxs) > 10 else idxs}")  # 最初の10個を表示
        
        print(f"Load {len(list_of_points)} data from {data_path}.")
        
        # カテゴリ名のマッピング（ORB-PointLLMの設定ファイルを使用）
        catfile = "/groups/gag51404/ide/ORB-PointLLM/pointllm/data/modelnet_config/modelnet40_shape_names_modified.txt"
        if os.path.exists(catfile):
            categories = [line.rstrip() for line in open(catfile)]
            print(f"🎯 Loaded {len(categories)} categories from ORB-PointLLM config")
        else:
            # フォールバック: Original PointLLMの設定ファイル
            catfile_orig = "/groups/gag51404/ide/PointLLM/pointllm/data/modelnet_config/modelnet40_shape_names_modified.txt"
            if os.path.exists(catfile_orig):
                categories = [line.rstrip() for line in open(catfile_orig)]
                print(f"🎯 Loaded {len(categories)} categories from Original PointLLM config")
            else:
                # 最終フォールバック: ハードコードされた基本的なカテゴリ名（Original PointLLMと一致）
                categories = [
                    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair",
                    "cone", "cup", "curtain", "desk", "door", "dresser", "flower pot", "glass box",
                    "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor", "night stand",
                    "person", "piano", "plant", "radio", "range hood", "sink", "sofa", "stairs",
                    "stool", "table", "tent", "toilet", "tv stand", "vase", "wardrobe", "xbox"
                ]
                print(f"🎯 Using fallback categories ({len(categories)} items)")
        
        # データセット風のオブジェクトを作成
        class CustomModelNetData:
            def __init__(self, points_list, labels_list, categories):
                self.list_of_points = points_list
                self.list_of_labels = labels_list
                self.categories = categories
                self.use_color = True
                self.normalize_pc = True
            
            def __len__(self):
                return len(self.list_of_labels)
            
            def pc_norm(self, pc):
                xyz = pc[:, :3]
                other_feature = pc[:, 3:]
                centroid = np.mean(xyz, axis=0)
                xyz = xyz - centroid
                m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
                xyz = xyz / m
                return np.concatenate((xyz, other_feature), axis=1)
            
            def __getitem__(self, index):
                points = self.list_of_points[index]
                label = self.list_of_labels[index]
                
                # 色情報を追加（6チャンネル対応）
                if points.shape[1] == 3:
                    # XYZのみの場合、RGB=0を追加
                    points = np.concatenate([points, np.zeros_like(points)], axis=1)
                
                # 正規化
                if self.normalize_pc:
                    points = self.pc_norm(points)
                
                points = torch.from_numpy(points).float()
                label_value = int(label.item()) if hasattr(label, 'item') else int(label)
                label_name = self.categories[label_value] if label_value < len(self.categories) else f"unknown_{label_value}"
                
                return {
                    "indice": index,
                    "point_clouds": points,
                    "labels": label,
                    "label_names": label_name
                }
        
        return CustomModelNetData(list_of_points, list_of_labels, categories)
    
    def _create_multi_cloud_samples(self, num_samples, use_sequential_sampling=False):
        """ModelNetからマルチクラウドサンプルを作成"""
        samples = []
        total_available = len(self.modelnet)
        
        # ランダムシードを固定（Original PointLLMと一致させる）
        import random
        random.seed(0)
        np.random.seed(0)
        print("🎯 Fixed random seed to 0 for consistency with Original PointLLM")
        
        # 順次サンプリングの場合（Original PointLLMとの比較用）
        if use_sequential_sampling:
            print(f"Using sequential sampling for comparison with Original PointLLM")
            
            # subset_numsが設定されている場合、Original PointLLMと同じサブセット選択を再現
            if hasattr(self.modelnet, 'list_of_points') and hasattr(self.modelnet, 'list_of_labels'):
                # Direct pickle loading case - need to replicate ModelNet subset selection
                import random
                random.seed(0)  # Original PointLLMと同じシード
                total_data_size = len(self.modelnet.list_of_points)
                if num_samples <= total_data_size:
                    # Original PointLLMの方法を再現: random.sample
                    selected_indices = random.sample(range(total_data_size), num_samples)
                    print(f"🎯 Sequential sampling with Original PointLLM indices: {selected_indices}")
                    
                    for i, original_idx in enumerate(selected_indices):
                        sample = self._create_sample_from_indices([i], i)  # Use sequential index as sample_id
                        # But track the original ModelNet index for debugging
                        sample['original_modelnet_index'] = original_idx
                        samples.append(sample)
                else:
                    # Fallback to simple sequential
                    for i in range(min(num_samples, total_available)):
                        sample = self._create_sample_from_indices([i], i)
                        samples.append(sample)
            else:
                # Regular dataset case
                for i in range(min(num_samples, total_available)):
                    # Single cloud (object identification)の場合
                    if self.num_clouds == 1:
                        sample = self._create_sample_from_indices([i], i)
                    else:
                        # Multi-cloudの場合でも、メインオブジェクトとしてindex iを使用
                        # 追加のクラウドはランダムサンプリング
                        additional_indices = []
                        for _ in range(self.num_clouds - 1):
                            # メインオブジェクトと異なるインデックスを選択
                            available_indices = [idx for idx in range(total_available) if idx != i and idx not in additional_indices]
                            if available_indices:
                                additional_indices.append(random.choice(available_indices))
                        
                        all_indices = [i] + additional_indices
                        sample = self._create_sample_from_indices(all_indices, i)
                    
                    samples.append(sample)
            
            print(f"Created {len(samples)} sequential samples for comparison")
            return samples
        
        # カテゴリ別のサンプルインデックスを取得（従来のランダムサンプリング）
        category_samples = {}
        for idx in range(total_available):
            data = self.modelnet[idx]
            label = data['labels']
            # numpy配列やtensorを適切に処理
            if hasattr(label, 'item'):
                label = label.item()
            elif isinstance(label, np.ndarray):
                label = int(label)
            else:
                label = int(label)
            
            if label not in category_samples:
                category_samples[label] = []
            category_samples[label].append(idx)
        
        available_categories = list(category_samples.keys())
        print(f"Available categories: {len(available_categories)}")
        
        for i in range(num_samples):
            sample = None
            
            if self.task_type == "shape_matching" and self.num_clouds == 2:
                # Shape Matchingの場合、同じカテゴリと異なるカテゴリを意図的に混在させる
                if i % 2 == 0 and len(available_categories) > 0:
                    # 偶数サンプル: 同じカテゴリのペアを作成
                    # 十分なサンプルがあるカテゴリを選択
                    suitable_categories = [cat for cat, indices in category_samples.items() 
                                         if len(indices) >= 2]
                    if suitable_categories:
                        selected_category = random.choice(suitable_categories)
                        selected_indices = random.sample(category_samples[selected_category], 2)
                        sample = self._create_sample_from_indices(selected_indices, i)
                        print(f"Sample {i}: Same category pair ({self.modelnet[selected_indices[0]]['label_names']})")
                
                if sample is None:
                    # 奇数サンプルまたは同じカテゴリのペアが作成できない場合: 異なるカテゴリのペア
                    if len(available_categories) >= 2:
                        # 異なる2つのカテゴリを選択
                        selected_categories = random.sample(available_categories, 2)
                        selected_indices = []
                        for cat in selected_categories:
                            selected_indices.append(random.choice(category_samples[cat]))
                        sample = self._create_sample_from_indices(selected_indices, i)
                        categories = [self.modelnet[idx]['label_names'] for idx in selected_indices]
                        print(f"Sample {i}: Different category pair ({categories[0]} vs {categories[1]})")
                    else:
                        # フォールバック: ランダムサンプリング
                        selected_indices = random.sample(range(total_available), min(2, total_available))
                        sample = self._create_sample_from_indices(selected_indices, i)
            
            if sample is None:
                # その他のタスクまたはフォールバック: ランダムサンプリング
                selected_indices = random.sample(range(total_available), min(self.num_clouds, total_available))
                sample = self._create_sample_from_indices(selected_indices, i)
            
            samples.append(sample)
        
        # 統計情報を表示
        if self.task_type == "shape_matching":
            same_count = sum(1 for s in samples if s.get('same_category', False))
            diff_count = len(samples) - same_count
            print(f"Shape matching samples - Same category: {same_count}, Different category: {diff_count}")
        
        return samples
    
    def _create_sample_from_indices(self, selected_indices, sample_id):
        """選択されたインデックスからサンプルを作成"""
        # 対応するデータを取得
        modelnet_data = [self.modelnet[idx] for idx in selected_indices]
        raw_labels = [data['labels'] for data in modelnet_data]
        label_names = [data['label_names'] for data in modelnet_data]
        
        # ラベルを適切に処理（numpy配列やtensorをintに変換）
        labels = []
        for label in raw_labels:
            if hasattr(label, 'item'):
                labels.append(label.item())
            elif isinstance(label, np.ndarray):
                labels.append(int(label))
            else:
                labels.append(int(label))
        
        # Sequential samplingの場合、メインオブジェクトのIDをOriginal PointLLMと合わせる
        if self.use_sequential_sampling and self.num_clouds == 1:
            # Single cloud comparison用：object_idを一致させる
            sample = {
                'indices': selected_indices,
                'labels': labels,
                'label_names': label_names,
                'task_type': self.task_type,
                'sample_id': selected_indices[0],  # メインオブジェクトのindexを使用
                'object_id': selected_indices[0]  # Original PointLLMとの比較用
            }
        else:
            sample = {
                'indices': selected_indices,
                'labels': labels,
                'label_names': label_names,
                'task_type': self.task_type,
                'sample_id': sample_id
            }
        
        # タスク固有のground truthを追加
        if self.task_type == "shape_matching":
            # 同じクラスかどうか
            sample['same_category'] = len(set(labels)) == 1
            sample['matching_result'] = "same" if sample['same_category'] else "different"
        elif self.task_type == "object_identification":
            # 各オブジェクトの正解ラベル
            sample['ground_truth_labels'] = label_names
        elif self.task_type == "geometric_reasoning":
            # 幾何学的関係の定義
            sample['relationship_type'] = self._analyze_geometric_relationship(labels)
        
        return sample
    
    def _analyze_geometric_relationship(self, labels):
        """幾何学的関係を分析"""
        if len(set(labels)) == 1:
            return "identical_category"
        else:
            return "different_categories"
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 複数の点群データを取得
        point_clouds = []
        for model_idx in sample['indices']:
            data = self.modelnet[model_idx]
            point_clouds.append(data['point_clouds'])
        
        return {
            'point_clouds_list': point_clouds,
            'sample_info': sample,
            'sample_id': idx
        }

def custom_collate_fn(batch):
    """
    カスタムcollate関数
    リスト構造を保持したままバッチ処理を行う
    """
    point_clouds_list = []
    sample_info = {}
    sample_ids = []
    
    # 各サンプルからデータを取得
    for item in batch:
        # 点群データ
        clouds = item['point_clouds_list']
        if not point_clouds_list:
            # 初回: 点群リストの構造を初期化
            point_clouds_list = [[] for _ in range(len(clouds))]
        
        for i, cloud in enumerate(clouds):
            point_clouds_list[i].append(cloud)
        
        # サンプル情報
        info = item['sample_info']
        for key, value in info.items():
            if key not in sample_info:
                sample_info[key] = []
            sample_info[key].append(value)
        
        # サンプルID
        sample_ids.append(item['sample_id'])
    
    # 点群データをテンソルに変換
    for i in range(len(point_clouds_list)):
        point_clouds_list[i] = torch.stack(point_clouds_list[i])
    
    return {
        'point_clouds_list': point_clouds_list,
        'sample_info': sample_info,
        'sample_id': sample_ids
    }

def init_model(args):
    """モデルの初期化"""
    print(f'[INFO] Loading model: {os.path.basename(args.model_name)}')
    
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # デバイスの自動選択
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.bfloat16
        print(f"Using CUDA device: {device}")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print(f"Using CPU device: {device}")
    
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_name, 
        low_cpu_mem_usage=True,
        use_cache=True, 
        torch_dtype=dtype,
    ).to(device)
    
    # デバッグ情報を表示
    print(f"Model class: {type(model).__name__}")
    print(f"Config class: {type(model.config).__name__}")
    print(f"Model file location: {model.__class__.__module__}")
    
    # マルチクラウド機能を強制的に有効化
    if hasattr(model.config, 'enable_multi_cloud'):
        print(f"Original multi-cloud setting: {model.config.enable_multi_cloud}")
        if args.force_enable_multi_cloud:
            model.config.enable_multi_cloud = True
            model.config.max_point_clouds = getattr(args, 'max_point_clouds', 8)
            model.config.use_segment_embedding = True
            model.config.use_local_position_embedding = True
            print(f"Multi-cloud functionality ENABLED: max_clouds={model.config.max_point_clouds}")
        else:
            print("Multi-cloud functionality disabled (use --force_enable_multi_cloud to enable)")
    else:
        print("Model does not support multi-cloud functionality")
    
    # トークナイザーの初期化
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    
    conv_mode = "vicuna_v1_1"
    conv = conv_templates[conv_mode].copy()
    
    return model, tokenizer, conv

def create_prompt(task_type, num_clouds, prompt_index=0, use_original_prompts=False):
    """プロンプトの作成"""
    if num_clouds == 1:
        if use_original_prompts:
            return ORIGINAL_IN_DISTRIBUTION_PROMPTS[prompt_index % len(ORIGINAL_IN_DISTRIBUTION_PROMPTS)]
        else:
            return SINGLE_CLOUD_PROMPTS[prompt_index % len(SINGLE_CLOUD_PROMPTS)]
    
    # Multi-cloud case
    if use_original_prompts:
        if task_type in ORIGINAL_MULTI_CLOUD_PROMPTS:
            prompts = ORIGINAL_MULTI_CLOUD_PROMPTS[task_type]
            return prompts[prompt_index % len(prompts)]
        else:
            # Original style fallback for multi-cloud
            cloud_tokens = " ".join([f"<cloud_{i}>" for i in range(num_clouds)])
            return f"What are these? {cloud_tokens}"
    else:
        if task_type in MULTI_CLOUD_PROMPTS:
            prompts = MULTI_CLOUD_PROMPTS[task_type]
            return prompts[prompt_index % len(prompts)]
        else:
            # デフォルトプロンプト
            cloud_tokens = " ".join([f"<cloud_{i}>" for i in range(num_clouds)])
            return f"Analyze these objects: {cloud_tokens}. Describe their relationship."

def generate_outputs(model, tokenizer, input_ids, point_clouds_data, stopping_criteria, 
                    do_sample=True, temperature=1.0, top_k=50, max_length=2048, top_p=0.95):
    """出力生成"""
    model.eval()
    with torch.inference_mode():
        # マルチクラウド対応かチェック
        if hasattr(model.config, 'enable_multi_cloud') and model.config.enable_multi_cloud and len(point_clouds_data) > 1:
            # マルチクラウドモード
            output_ids = model.generate(
                input_ids,
                point_clouds_list=point_clouds_data,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                max_length=max_length,
                top_p=top_p,
                stopping_criteria=[stopping_criteria]
            )
        else:
            # シングルクラウドモード（最初の点群のみ使用）
            first_cloud = point_clouds_data[0] if point_clouds_data else None
            if first_cloud is not None:
                first_cloud = first_cloud.unsqueeze(0)  # バッチ次元を追加
            
            output_ids = model.generate(
                input_ids,
                point_clouds=first_cloud,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                max_length=max_length,
                top_p=top_p,
                stopping_criteria=[stopping_criteria]
            )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]
    
    torch.cuda.empty_cache()
    return outputs

def start_evaluation(model, tokenizer, conv, dataloader, task_type, prompt_index, output_dir, output_file, use_original_prompts=False):
    """評価の実行"""
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    
    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    mm_use_point_start_end = point_backbone_config.get('mm_use_point_start_end', False)
    
    # マルチクラウド用トークンの確認
    has_multi_cloud = hasattr(model.config, 'enable_multi_cloud') and model.config.enable_multi_cloud
    cloud_token_ids = point_backbone_config.get('cloud_token_ids', {}) if has_multi_cloud else {}
    
    # プロンプトテンプレートの例（表示用）
    example_prompt = create_prompt(task_type, 2 if task_type != "object_identification" else 1, prompt_index, use_original_prompts)
    
    results = {
        "task_type": task_type,
        "prompt_index": prompt_index,
        "prompt_template": example_prompt,
        "model_config": {
            "enable_multi_cloud": has_multi_cloud,
            "max_point_clouds": getattr(model.config, 'max_point_clouds', 8) if has_multi_cloud else 1,
            "point_token_len": point_token_len
        },
        "results": []
    }
    
    print(f"Multi-cloud support: {has_multi_cloud}")
    print(f"Available cloud tokens: {list(cloud_token_ids.keys()) if cloud_token_ids else 'None'}")
    
    for batch in tqdm(dataloader, desc=f"Evaluating {task_type}"):
        point_clouds_list = batch["point_clouds_list"]
        sample_info = batch["sample_info"]
        sample_ids = batch["sample_id"]
        
        batch_size = len(point_clouds_list[0])
        num_clouds = len(point_clouds_list)
        
        for b in range(batch_size):
            # 現在のサンプルの点群データを準備
            device = next(model.parameters()).device
            current_clouds = [pc[b].to(device).to(model.dtype) for pc in point_clouds_list]
            
            # sample_info の適切な処理
            sample_info_item = {}
            for k, v in sample_info.items():
                if torch.is_tensor(v[b]):
                    sample_info_item[k] = v[b].item()
                elif isinstance(v[b], list):
                    sample_info_item[k] = v[b]  # リストはそのまま保持
                else:
                    sample_info_item[k] = v[b]
            
            # プロンプトの作成
            original_prompt = create_prompt(task_type, num_clouds, prompt_index, use_original_prompts)
            qs = original_prompt  # モデル用のプロンプト
            
            # マルチクラウドトークンの処理（モデル用）
            if num_clouds > 1 and has_multi_cloud:
                for i in range(num_clouds):
                    cloud_token = f'<cloud_{i}>'
                    if cloud_token in qs:
                        # 実際のトークンIDが利用可能な場合
                        if mm_use_point_start_end:
                            replace_token = (
                                point_backbone_config.get('default_point_start_token', '<point_start>') + 
                                default_point_patch_token * point_token_len + 
                                point_backbone_config.get('default_point_end_token', '<point_end>')
                            )
                        else:
                            replace_token = default_point_patch_token * point_token_len
                        
                        qs = qs.replace(cloud_token, replace_token)
            elif num_clouds == 1:
                # 単一点群の場合
                if mm_use_point_start_end:
                    point_token = (
                        point_backbone_config.get('default_point_start_token', '<point_start>') + 
                        default_point_patch_token * point_token_len + 
                        point_backbone_config.get('default_point_end_token', '<point_end>')
                    )
                else:
                    point_token = default_point_patch_token * point_token_len
                
                qs = point_token + '\n' + qs
            
            # 会話の設定
            conv_copy = conv.copy()
            conv_copy.append_message(conv_copy.roles[0], qs)
            conv_copy.append_message(conv_copy.roles[1], None)
            
            prompt = conv_copy.get_prompt()
            inputs = tokenizer([prompt])
            input_ids = torch.as_tensor(inputs.input_ids).to(device)
            
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
            
            # 出力生成
            try:
                outputs = generate_outputs(
                    model, tokenizer, input_ids, current_clouds, stopping_criteria
                )
                model_output = outputs[0] if outputs else ""
            except Exception as e:
                print(f"Error generating output for sample {sample_ids[b]}: {e}")
                model_output = f"Error: {str(e)}"
            
            # 結果の保存（元のプロンプトを保存、トークン化されたものではない）
            display_prompt = original_prompt
            if num_clouds > 1 and has_multi_cloud:
                # 表示用にクラウドトークンを分かりやすい形式に置換
                for i in range(num_clouds):
                    cloud_token = f'<cloud_{i}>'
                    if cloud_token in display_prompt:
                        display_prompt = display_prompt.replace(cloud_token, f'[Point Cloud {i+1}]')
            elif num_clouds == 1:
                display_prompt = '[Point Cloud]\n' + original_prompt
            
            result = {
                "sample_id": sample_ids[b].item() if torch.is_tensor(sample_ids[b]) else sample_ids[b],
                "num_clouds": num_clouds,
                "task_type": task_type,
                "prompt": display_prompt,
                "model_output": model_output,
                "ground_truth": sample_info_item
            }
            
            results["results"].append(result)
    
    # 結果の保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    
    print(f"Saved {len(results['results'])} results to {output_path}")
    
    # 簡単な統計を表示
    _print_evaluation_stats(results)
    
    return results

def _print_evaluation_stats(results):
    """評価統計の表示"""
    if not results.get("results"):
        return
    
    print(f"\n=== Evaluation Summary ===")
    print(f"Task: {results['task_type']}")
    print(f"Total samples: {len(results['results'])}")
    
    # タスク固有の統計
    task_results = results['results']
    if results['task_type'] == "shape_matching":
        same_category_results = [r for r in task_results 
                               if r['ground_truth'].get('same_category', False)]
        diff_category_results = [r for r in task_results 
                               if not r['ground_truth'].get('same_category', False)]
        
        print(f"Same category pairs: {len(same_category_results)}/{len(task_results)}")
        print(f"Different category pairs: {len(diff_category_results)}/{len(task_results)}")
        
        # 同じカテゴリペアの詳細
        if same_category_results:
            print("\nSame category pairs:")
            for i, r in enumerate(same_category_results[:3]):
                gt = r['ground_truth']
                indices = gt.get('indices', [])
                label_names = gt.get('label_names', [])
                model_output = r['model_output'][:100] + "..." if len(r['model_output']) > 100 else r['model_output']
                print(f"  Sample {r['sample_id']}: {label_names[0]} vs {label_names[1]}")
                print(f"    Model says: {model_output}")
        
        # 異なるカテゴリペアの詳細
        if diff_category_results:
            print("\nDifferent category pairs:")
            for i, r in enumerate(diff_category_results[:3]):
                gt = r['ground_truth']
                indices = gt.get('indices', [])
                label_names = gt.get('label_names', [])
                model_output = r['model_output'][:100] + "..." if len(r['model_output']) > 100 else r['model_output']
                print(f"  Sample {r['sample_id']}: {label_names[0]} vs {label_names[1]}")
                print(f"    Model says: {model_output}")
            
    elif results['task_type'] == "object_identification":
        num_clouds = task_results[0]['num_clouds'] if task_results else 0
        print(f"Objects per sample: {num_clouds}")
        
        # 詳細表示
        print("\nSample objects:")
        for i, r in enumerate(task_results[:3]):  # 最初の3サンプルを表示
            gt = r['ground_truth']
            indices = gt.get('indices', [])
            label_names = gt.get('label_names', [])
            model_output = r['model_output'][:80] + "..." if len(r['model_output']) > 80 else r['model_output']
            print(f"  Sample {r['sample_id']}: {label_names}")
            print(f"    Model says: {model_output}")
    
    print("=========================\n")

def main(args):
    # 引数の処理
    model_basename = os.path.basename(os.path.expanduser(args.model_name))
    
    if args.output_dir is None:
        args.output_dir = os.path.join("evaluation_results", model_basename)
    
    # 出力ファイル名の作成
    suffix = f"{args.num_clouds}clouds" if args.num_clouds > 1 else "single"
    prompt_type = "original" if args.use_original_prompts else "custom"
    args.output_file = f"modelnet_{args.task_type}_{suffix}_{prompt_type}_prompt{args.prompt_index}.json"
    args.output_file_path = os.path.join(args.output_dir, args.output_file)
    
    print(f"Data path: {args.data_path}")
    print(f"Output: {args.output_file_path}")
    
    # 結果生成または読み込み
    if not os.path.exists(args.output_file_path) or args.force_regenerate:
        # データセットの作成
        dataset = ModelNetMultiCloudDataset(
            data_path=args.data_path,
            task_type=args.task_type,
            num_clouds=args.num_clouds,
            num_samples=args.num_samples,
            pointnum=args.pointnum,
            use_color=args.use_color,
            subset_nums=args.subset_nums,
            use_sequential_sampling=args.use_sequential_sampling
        )
        
        # データローダーの作成
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, 
                              collate_fn=custom_collate_fn)
        
        # モデルの初期化
        model, tokenizer, conv = init_model(args)
        
        print(f'[INFO] Starting evaluation: {args.output_file}')
        results = start_evaluation(
            model, tokenizer, conv, dataloader, args.task_type, 
            args.prompt_index, args.output_dir, args.output_file, args.use_original_prompts
        )
        
        # メモリ解放
        del model
        del tokenizer
        torch.cuda.empty_cache()
    else:
        print(f'[INFO] Loading existing results: {args.output_file_path}')
        with open(args.output_file_path, 'r') as fp:
            results = json.load(fp)
        
        _print_evaluation_stats(results)
    
    print(f"Evaluation completed. Results: {args.output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Cloud ModelNet40 Evaluation")
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="RunsenXu/PointLLM_7B_v1.2",
                        help="Path to the model")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    
    # Dataset settings
    parser.add_argument("--data_path", type=str, 
                        default="/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat",
                        help="Path to ModelNet40 .dat file")
    parser.add_argument("--use_color", action="store_true", default=True,
                        help="Use color information (forced True for PointLLM compatibility)")
    parser.add_argument("--pointnum", type=int, default=8192,
                        help="Number of points per cloud (fixed at 8192 for PointLLM)")
    parser.add_argument("--subset_nums", type=int, default=-1,
                        help="Subset size (-1 for all data)")
    
    # Multi-cloud settings
    parser.add_argument("--task_type", type=str, default="shape_matching",
                        choices=["shape_matching", "part_assembly", "geometric_reasoning", "object_identification"],
                        help="Type of multi-cloud task")
    parser.add_argument("--num_clouds", type=int, default=2,
                        help="Number of point clouds per sample")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of samples to evaluate")
    
    # Generation settings
    parser.add_argument("--prompt_index", type=int, default=0,
                        help="Index of prompt template to use")
    parser.add_argument("--use_original_prompts", action="store_true", default=False,
                        help="Use original PointLLM in-distribution prompts (e.g., 'What is this?')")
    parser.add_argument("--force_regenerate", action="store_true", default=False,
                        help="Force regenerate results even if file exists")
    
    # Multi-cloud model settings
    parser.add_argument("--force_enable_multi_cloud", action="store_true", default=False,
                        help="Force enable multi-cloud functionality in the model")
    parser.add_argument("--max_point_clouds", type=int, default=8,
                        help="Maximum number of point clouds supported")
    
    # Comparison settings
    parser.add_argument("--use_sequential_sampling", action="store_true", default=False,
                        help="Use sequential sampling to match Original PointLLM object_ids for comparison")
    
    args = parser.parse_args()
    
    # ランダムシードの設定
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    main(args) 