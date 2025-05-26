# Multi-GPU Parallelization for 3D Gaussian Splatting

## Project Overview
This project implements and optimizes multi-GPU parallelization for 3D Gaussian Splatting (3DGS), a state-of-the-art 3D scene representation and rendering technique. The implementation focuses on efficient distributed training across multiple GPUs, with comprehensive profiling and performance analysis.

## Key Features
- Multi-GPU distributed training implementation
- Performance profiling tools for both Python and CUDA operations
- Optimized data loading and processing pipeline
- Comprehensive benchmarking suite
- Support for various GPU configurations (A100, V100)

## Project Structure
```
.
├── setup_env/           # Environment setup scripts and documentation
├── modified/           # Modified core implementation files
├── profiling/          # Profiling tools and scripts
│   ├── cuda_profile.sh    # CUDA profiling script
│   ├── python_profile.sh  # Python profiling script
│   └── run_profiler.py    # Profiling orchestration
├── output/             # Training outputs and results
└── imgs/              # Project documentation images
```

## Prerequisites
- CUDA-compatible GPUs (tested on A100 and V100)
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+

## Installation

1. Clone the original implementation:
```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```

2. Replace core files with our optimized versions from the `modified/` directory

3. Set up the environment:
```bash
# Follow instructions in ./setup_env/README.md
```

## Usage

### Multi-GPU Training
Request a GPU session:
```bash
# For 2 GPUs, 20-minute session
srun --partition=dpart --qos=medium --gres=gpu:2 --time 0:20:0 --pty bash
# OR for A100 GPUs
srun --partition=dpart --qos=normal --gres=gpu:a100:2 --time 0:20:0 --pty bash
```

Run distributed training:
```bash
# Train with 2 GPUs, 1/8 resolution, 200 iterations
torchrun --nproc_per_node=2 train.py -s ./bicycle/ --test_iterations 200 --iterations 200 -r 8
```

### Performance Profiling

#### Python Profiling
```bash
# Step 1: Run profiling
python -m cProfile -o profiling_results/profile_train.prof train.py -s data/bicycle/ --iterations 200

# Step 2: Analyze results
python pyProfile.py
```

#### CUDA Profiling
```bash
# Using our custom CUDA profiler
python train_cudaProfile.py -s ./bicycle/ --iterations 200
```

## Dataset
We use the RefRaw360 dataset:
- Source: https://storage.googleapis.com/gresearch/refraw360/360_v2.zip
- Resolution: 1/8 scale images from `360_v2/bicycle/images_8`

## Performance Optimization
Our implementation includes several optimizations:
1. Efficient data distribution across GPUs
2. Optimized memory management
3. Custom CUDA kernels for parallel processing
4. Reduced communication overhead between GPUs

## Acknowledgments
- Original 3D Gaussian Splatting implementation by [graphdeco-inria](https://github.com/graphdeco-inria/gaussian-splatting)
- CMSC714 Spring 2025 Course Project