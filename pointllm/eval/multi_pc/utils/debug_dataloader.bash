#!/bin/bash

# Debug DataLoader sizes
# 通常のPointLLMとORB-PointLLMのDataLoaderサイズを確認

echo "=== DataLoader Debug Test ==="

DATA_PATH="/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat"
MODEL_NAME="RunsenXu/PointLLM_7B_v1.2"

echo ""
echo "1. Original PointLLM DataLoader size check..."

python - << 'EOF'
import sys
import os
sys.path.append('/groups/gag51404/ide/PointLLM')

from pointllm.data import ModelNet
from torch.utils.data import DataLoader

# Test original PointLLM dataset loading
data_path = "/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat"

print("Testing with subset_nums=20, batch_size=1:")
dataset = ModelNet(config_path=None, split="test", subset_nums=20, use_color=True, data_path=data_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

print(f"Dataset size: {len(dataset)}")
print(f"DataLoader batches: {len(dataloader)}")
print(f"Expected progress: {len(dataloader)} batches")

# Test first few batches
print("\nFirst 3 batch info:")
for i, batch in enumerate(dataloader):
    if i >= 3:
        break
    print(f"  Batch {i}: point_clouds shape = {batch['point_clouds'].shape}, label = {batch['label_names']}")

print("Original PointLLM dataset test complete.")
EOF

echo ""
echo "2. ORB-PointLLM DataLoader size check..."

python - << 'EOF'
import sys
import os
sys.path.insert(0, '/groups/gag51404/ide/ORB-PointLLM')

from pointllm.eval.eval_multi_cloud import ModelNetMultiCloudDataset, custom_collate_fn
from torch.utils.data import DataLoader

# Test ORB-PointLLM dataset loading
data_path = "/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat"

print("Testing with num_samples=5, subset_nums=20:")
dataset = ModelNetMultiCloudDataset(
    data_path=data_path,
    task_type="object_identification",
    num_clouds=1,
    num_samples=5,
    pointnum=8192,
    use_color=True,
    subset_nums=20
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

print(f"Dataset size: {len(dataset)}")
print(f"DataLoader batches: {len(dataloader)}")
print(f"Expected progress: {len(dataloader)} batches")

# Test first few batches
print("\nFirst 3 batch info:")
for i, batch in enumerate(dataloader):
    if i >= 3:
        break
    print(f"  Batch {i}: num_clouds = {len(batch['point_clouds_list'])}, sample_id = {batch['sample_id']}")

print("ORB-PointLLM dataset test complete.")
EOF

echo ""
echo "=== Debug Complete ==="
echo "Check if dataset sizes and DataLoader batches match expectations." 