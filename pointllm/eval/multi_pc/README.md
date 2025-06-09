# ORB-PointLLM Multi-Cloud Evaluation Suite

Complete evaluation suite for testing ORB-PointLLM's multi-cloud functionality and comparing with Original PointLLM.

## 🎯 Quick Start - Comparison with Original PointLLM

**To run comparison between Original PointLLM and ORB-PointLLM:**

```bash
cd comparison
./run_comparison.bash
```

This is the **main script** you need for comparing implementations!

## 📁 Directory Structure

```
multi_pc/
├── 🔧 comparison/                     # MAIN: Original vs ORB-PointLLM comparison
│   ├── run_comparison.bash            # 🎯 PRIMARY SCRIPT - Run this!
│   ├── README.md                      # Detailed comparison guide
│   └── comparison_results/            # Generated comparison results
├── 📁 core/                           # Core evaluation engine
│   ├── eval_multi_cloud.py            # Main evaluation script
│   └── README.md                      # Core functionality docs
├── 📁 basic_tests/                    # Basic functionality tests
│   ├── quick_test.bash                # 5-minute basic test
│   ├── gpu_test.bash                  # GPU functionality test
│   └── multi_test.bash                # Multi-task test
├── 📁 original_prompts/               # Original PointLLM prompt testing
│   ├── test_original_prompts.bash     # Comprehensive original prompt test
│   └── quick_original_test.bash       # Quick original prompt test
├── 📁 utils/                          # Debug utilities
│   └── debug_dataloader.bash          # Dataloader debugging
├── 📁 docs/                           # Documentation
└── 📁 initial_test/                   # Initial development tests
```

## 🚀 Usage Scenarios

### 1. 🎯 Compare Original PointLLM vs ORB-PointLLM (RECOMMENDED)

```bash
cd comparison
./run_comparison.bash          # Default: 20 samples
./run_comparison.bash 5 5      # Quick test: 5 samples  
./run_comparison.bash 50 50    # Comprehensive: 50 samples
```

**What it does:**
- Runs both Original PointLLM and ORB-PointLLM with identical conditions
- Uses sequential sampling for fair comparison
- Analyzes object_id consistency and model outputs
- Provides comprehensive comparison report

### 2. 🧪 Test Basic Functionality

```bash
cd basic_tests
./quick_test.bash              # 5-minute basic functionality test
./gpu_test.bash                # Test GPU compatibility
./multi_test.bash              # Test multiple tasks
```

### 3. 📝 Test Original PointLLM Prompts

```bash
cd original_prompts
./quick_original_test.bash     # Quick test with original prompts
./test_original_prompts.bash   # Comprehensive original prompt test
```

### 4. 🔧 Debug Issues

```bash
cd utils
./debug_dataloader.bash        # Debug data loading issues
```

## 📊 Key Features

### Sequential Sampling for Comparison
- **Purpose**: Ensures object_id consistency between implementations
- **Usage**: `--use_sequential_sampling` flag
- **Result**: Identical object sequences for fair comparison

### Original Prompt Support
- **Prompts**: "What is this?", "This is an object of"
- **Usage**: `--use_original_prompts` flag
- **Purpose**: Test compatibility with Original PointLLM training data

### Multi-Cloud Tasks
- **object_identification**: Single cloud object recognition
- **shape_matching**: Compare multiple objects
- **part_assembly**: Analyze object relationships
- **geometric_reasoning**: Spatial relationship analysis

## 🎉 Success Indicators

### Comparison Success
```
🎯 Object Matching Rate: 20/20 (100.0%)
🏆 EXCELLENT: Results are highly comparable!
✅ Implementation consistency verified
```

### Basic Test Success
```
✅ Single cloud evaluation: PASSED
✅ Multi-cloud evaluation: PASSED  
✅ Model loading: PASSED
✅ Data processing: PASSED
```

## 🛠️ Troubleshooting

### CUDA Environment Issues
```bash
# Check GPU availability
nvidia-smi

# Set specific GPU
export CUDA_VISIBLE_DEVICES=0

# If CUDA unavailable, scripts will use CPU mode
```

### Path Issues
```bash
# Always run from the appropriate directory:
cd /groups/gag51404/ide/ORB-PointLLM/pointllm/eval/multi_pc/[subdirectory]
```

## 📋 Dependencies

- **Model**: `RunsenXu/PointLLM_7B_v1.2`
- **Data**: `/groups/gag51404/ide/PointLLM/data/modelnet40_data/modelnet40_test_8192pts_fps.dat`
- **Environment**: PointLLM conda environment with transformers, torch, etc.

## 🎯 Recommended Workflow

1. **Start with basic test**: `basic_tests/quick_test.bash`
2. **Run comparison**: `comparison/run_comparison.bash` 
3. **Test original prompts**: `original_prompts/quick_original_test.bash`
4. **Debug if needed**: `utils/debug_dataloader.bash`

## 📈 Expected Performance

- **Basic test**: ~5 minutes
- **Comparison (20 samples)**: ~10-15 minutes
- **Comprehensive test (50 samples)**: ~30-40 minutes

The evaluation suite ensures ORB-PointLLM maintains full compatibility with Original PointLLM while providing multi-cloud capabilities for advanced point cloud analysis. 