import os
import json
import torch
import numpy as np

import copy
import transformers
from torch.utils.data import Dataset

from .utils import *


def make_object_point_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for Joint3Ddataset with text and point cloud data."""
    """Initialize datasets."""

    # ===== 新規: マルチクラウド対応のコレーター =====
    if getattr(data_args, 'enable_multi_cloud', False):
        data_collator = MultiCloudDataCollatorForPointTextDataset(tokenizer=tokenizer)
        print("Using MultiCloudDataCollator for multi-cloud training")
    else:
    data_collator = DataCollatorForPointTextDataset(tokenizer=tokenizer)
        print("Using standard DataCollator")

    if data_args.split_train_val:
        print("Loading training datasets.")
        train_dataset = ObjectPointCloudDataset(
            split='train',
            data_path=data_args.data_path,
            anno_path=data_args.anno_path,
            pointnum=data_args.pointnum,
            conversation_types=data_args.conversation_types,
            tokenizer=tokenizer,
            use_color=data_args.use_color,
            data_args=data_args
        )
        print("Done!")
        if data_args.data_debug_num > 0:
            print('Debug mode, using training set as val set.')
            val_dataset = train_dataset
        else:
            print("Loading validation datasets.")
            val_dataset = ObjectPointCloudDataset(
                split='val',
                data_path=data_args.data_path,
                anno_path=data_args.anno_path,
                pointnum=data_args.pointnum,
                conversation_types=data_args.conversation_types,
                tokenizer=tokenizer,
                use_color=data_args.use_color,
                data_args=data_args
            )
        return dict(train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator)
    else:
        train_dataset = ObjectPointCloudDataset(
            split='train',
            data_path=data_args.data_path,
            anno_path=data_args.anno_path,
            pointnum=data_args.pointnum,
            conversation_types=data_args.conversation_types,
            use_color=data_args.use_color,
            tokenizer=tokenizer,
            data_args=data_args
        )
        return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

class ObjectPointCloudDataset(Dataset):
    """Dataset utilities for objaverse with multi-cloud support."""
    def __init__(self,
                 data_path=None,
                 anno_path=None,
                 tokenizer=None,
                 pointnum=8192,
                 split='train',
                 conversation_types=None,
                 use_color=True,
                 data_args=None):

        super(ObjectPointCloudDataset, self).__init__()

        self.data_path = data_path
        self.anno_path = anno_path
        self.tokenizer = tokenizer
        self.split = split 
        if conversation_types is None:
            self.conversation_types = ("simple_description",)
        else:
            self.conversation_types = conversation_types

        self.data_args = data_args
        self.normalize_pc = True
        self.use_color = use_color

        self.pointnum = pointnum
        self.point_backbone_config = data_args.point_backbone_config if data_args is not None else None
        self.point_indicator = '<point>'

        # ===== 新規: マルチクラウド設定 =====
        self.enable_multi_cloud = getattr(data_args, 'enable_multi_cloud', False)
        self.max_point_clouds = getattr(data_args, 'max_point_clouds', 8)
        
        if self.enable_multi_cloud:
            print(f"Multi-cloud mode enabled: max_clouds={self.max_point_clouds}")
            # マルチクラウド用のインジケーター（複数の点群を示す）
            self.multi_cloud_indicators = [f'<cloud_{i}>' for i in range(self.max_point_clouds)]

        # Load the data list from JSON
        print(f"Loading anno file from {anno_path}.")
        with open(anno_path, "r") as json_file:
            self.list_data_dict = json.load(json_file)
        
        print(f"Using conversation_type: {self.conversation_types}") 
        print(f"Before filtering, the dataset size is: {len(self.list_data_dict)}.")

        filter_ids = ['6760e543e1d645d5aaacd3803bcae524', 'b91c0711149d460a8004f9c06d3b7f38'] if self.use_color else []

        # ===== 新規: マルチクラウドデータのフィルタリング =====
        self.list_data_dict = [
            data for data in self.list_data_dict 
            if data.get('conversation_type', 'simple_description') in self.conversation_types 
            and data.get('object_id') not in filter_ids
            and self._is_valid_multi_cloud_data(data)
        ]

        print(f"After filtering, the dataset size is: {len(self.list_data_dict)}.")
        for conversation_type in self.conversation_types:
            count = len([data for data in self.list_data_dict if data.get('conversation_type', 'simple_description') == conversation_type])
            print(f"Number of {conversation_type}: {count}")

        if self.data_args is not None and self.data_args.data_debug_num > 0:
            self.list_data_dict = self.list_data_dict[:self.data_args.data_debug_num]
            print('Debug mode, using: ' + ' '.join([data['object_id'] for data in self.list_data_dict]))
        elif self.data_args is not None and self.data_args.split_train_val:
            if self.split == 'train':
                self.list_data_dict = self.list_data_dict[:int(self.data_args.split_ratio * len(self.list_data_dict))]
                print(f"Train set size: {len(self.list_data_dict)}")
            else:
                self.list_data_dict = self.list_data_dict[int(self.data_args.split_ratio * len(self.list_data_dict)):]
                print(f"Val set size: {len(self.list_data_dict)}")

    def _is_valid_multi_cloud_data(self, data):
        """マルチクラウドデータの有効性をチェック"""
        if not self.enable_multi_cloud:
            return True
        
        # マルチクラウド用の会話データをチェック
        conversations = data.get('conversations', [])
        if not conversations:
            return True
        
        first_conversation = conversations[0].get('value', '')
        
        # マルチクラウドインジケーターの存在チェック
        multi_cloud_count = sum(1 for indicator in self.multi_cloud_indicators if indicator in first_conversation)
        
        # 単一点群の場合は常に有効
        if multi_cloud_count <= 1:
            return True
        
        # マルチクラウドの場合、対応するオブジェクトIDが存在するかチェック
        object_ids = data.get('object_ids', [data.get('object_id')])
        return len(object_ids) >= multi_cloud_count

    def _load_point_cloud(self, object_id, type='objaverse'):
        if type == 'objaverse':
            return self._load_objaverse_point_cloud(object_id) 

    def _load_objaverse_point_cloud(self, object_id):
        filename = f"{object_id}_{self.pointnum}.npy"
        point_cloud = np.load(os.path.join(self.data_path, filename))

        if not self.use_color:
            point_cloud = point_cloud[:, :3]

        return point_cloud

    def _load_multiple_point_clouds(self, object_ids):
        """複数の点群を読み込み"""
        point_clouds = []
        for object_id in object_ids:
            if object_id is not None:
                try:
                    point_cloud = self._load_point_cloud(object_id)
                    if self.normalize_pc:
                        point_cloud = self.pc_norm(point_cloud)
                    point_clouds.append(point_cloud)
                except Exception as e:
                    print(f"Error loading point cloud {object_id}: {e}")
                    point_clouds.append(None)
            else:
                point_clouds.append(None)
        return point_clouds

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        xyz = pc[:, :3]
        other_feature = pc[:, 3:]

        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        xyz = xyz / m

        pc = np.concatenate((xyz, other_feature), axis=1)
        return pc
    
    def _get_multi_cloud_item(self, index):
        """マルチクラウド用のデータ取得"""
        data_item = self.list_data_dict[index]
        sources = [data_item]
        
        # 複数のオブジェクトIDを取得
        object_ids = data_item.get('object_ids', [data_item.get('object_id')])
        
        # 会話データの前処理
        conversations = data_item['conversations']
        first_conversation = conversations[0]['value']
        
        # マルチクラウドインジケーターが含まれているかチェック
        multi_cloud_indicators_found = [
            indicator for indicator in self.multi_cloud_indicators 
            if indicator in first_conversation
        ]
        
        if len(multi_cloud_indicators_found) > 1:
            # 複数点群の場合
            point_clouds_list = self._load_multiple_point_clouds(object_ids[:len(multi_cloud_indicators_found)])
            
            if self.tokenizer is not None:
                sources = preprocess_multimodal_multi_cloud(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.point_backbone_config,
                    multi_cloud_indicators=multi_cloud_indicators_found
                )
            
            return sources, point_clouds_list, True
        else:
            # 単一点群の場合（既存の処理）
            object_id = object_ids[0] if object_ids else data_item.get('object_id')
            point_cloud = self._load_point_cloud(object_id)
            if self.normalize_pc:
                point_cloud = self.pc_norm(point_cloud)
            
            if self.tokenizer is not None and self.point_indicator in first_conversation:
                sources = preprocess_multimodal_point_cloud(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.point_backbone_config,
                    point_indicator=self.point_indicator
                )
            
            return sources, [point_cloud], False
    
    def __getitem__(self, index):
        if self.enable_multi_cloud:
            return self._get_multi_cloud_item_processed(index)
        else:
            return self._get_single_cloud_item(index)
    
    def _get_multi_cloud_item_processed(self, index):
        """マルチクラウド対応の完全なアイテム処理"""
        sources, point_clouds_data, is_multi_cloud = self._get_multi_cloud_item(index)
        
        if self.tokenizer is None:
            if is_multi_cloud:
                data_dict = dict(
                    point_clouds_list=[torch.from_numpy(pc.astype(np.float32)) if pc is not None else None for pc in point_clouds_data],
                    object_ids=self.list_data_dict[index].get('object_ids', [self.list_data_dict[index].get('object_id')]),
                    is_multi_cloud=is_multi_cloud
                )
            else:
                data_dict = dict(
                    point_clouds=torch.from_numpy(point_clouds_data[0].astype(np.float32)),
                    object_ids=self.list_data_dict[index].get('object_id'),
                    is_multi_cloud=is_multi_cloud
                )
            return data_dict

        data_dict = preprocess_v1(sources, self.tokenizer)
        
        if isinstance(index, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0]
            )

        # 点群データの追加
        if is_multi_cloud:
            valid_point_clouds = [torch.from_numpy(pc.astype(np.float32)) if pc is not None else None for pc in point_clouds_data]
            data_dict['point_clouds_list'] = valid_point_clouds
            data_dict['is_multi_cloud'] = True
        else:
            if point_clouds_data and point_clouds_data[0] is not None:
                data_dict['point_clouds'] = torch.from_numpy(point_clouds_data[0].astype(np.float32))
            data_dict['is_multi_cloud'] = False

        return data_dict
    
    def _get_single_cloud_item(self, index):
        """既存の単一点群処理（後方互換性）"""
        sources = self.list_data_dict[index]
        if isinstance(index, int):
            sources = [sources]
        assert len(sources) == 1, "sources should be a list"
        
        if self.point_indicator in sources[0]['conversations'][0]['value']:
            object_id = self.list_data_dict[index]['object_id']
            point_cloud = self._load_point_cloud(object_id)
            if self.normalize_pc:
                point_cloud = self.pc_norm(point_cloud)

            if self.tokenizer is None:
                data_dict = dict(
                    point_clouds=torch.from_numpy(point_cloud.astype(np.float32)),
                    object_ids=object_id
                )
                return data_dict

            sources = preprocess_multimodal_point_cloud(
                copy.deepcopy([e["conversations"] for e in sources]), 
                self.point_backbone_config, 
                point_indicator=self.point_indicator
            )
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess_v1(sources, self.tokenizer)

        if isinstance(index, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0]
            )

        # point exist in the data
        if self.point_indicator in self.list_data_dict[index]['conversations'][0]['value']:
            data_dict['point_clouds'] = torch.from_numpy(point_cloud.astype(np.float32))

        return data_dict

    def __len__(self):
        """Return number of utterances."""
        return len(self.list_data_dict)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="data/objaverse_data", type=str,
                        help="Path to the data directory.")
    parser.add_argument("--anno_path", default=None, type=str, required=True,
                        help="Path to the annotation file.")
    parser.add_argument("--split", default='train', type=str, 
                        help="Whether to use the train or validation dataset.")
    parser.add_argument("--pointnum", default=8192, type=int,
                        help="Number of points in the point cloud.")
    parser.add_argument("--data_debug_num", default=0, type=int,
                        help="Number of data to debug with.")
    parser.add_argument("--split_train_val", default=False, type=bool,
                        help="Whether to split the dataset into training and validation.")
    parser.add_argument("--split_ratio", default=0.9, type=float,
                        help="The ratio of training to validation data.")
    parser.add_argument("--tokenizer_path", default=None, type=str, required=True,
                        help="Path to the tokenizer config file.")
    
    args = parser.parse_args()

    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)

    args.point_backbone_config = None

    # Initialize dataset
    dataset = ObjectPointCloudDataset(
        data_path=args.data_path,
        anno_path=args.anno_path,
        pointnum=args.pointnum,
        split=args.split,
        tokenizer=tokenizer,
        data_args=args
    )

    # Example usage
    print(f'Dataset length: {len(dataset)}')

