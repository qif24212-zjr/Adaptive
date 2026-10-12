# Adaptive Frame Pruning with Temporal Context for Video Captioning 🚀

Our method adaptively prunes redundant video frames for efficient video captioning by:

1. uniformly sampling a pool of candidate frames from the input video;
2. estimating each candidate frame's contextual contribution based on its temporal relationships with neighboring frames and its relevance to a learnable task query;
3. selecting the top-\(K\) informative frames and feeding their visual tokens into the original captioning model.

The frame pruning module is trained end-to-end using the captioning loss, without requiring explicit frame-level annotations.
# 1. Installation ⚙️

First, clone the repository:

```bash
git clone https://github.com/qif24212-zjr/Adaptive.git
cd Adaptive
```

Then create and activate the Conda environment:

```bash
conda create -n adaptive-frame-pruning python=3.10 -y
conda activate adaptive-frame-pruning
```

Install PyTorch with CUDA 12.1 support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining dependencies according to the project configuration.

**Note:** A compatible NVIDIA driver is required to run PyTorch with CUDA support. You can verify the installation with:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

## Potential Issues

1. Missing packages: If you encounter `ModuleNotFoundError`, make sure all required dependencies are installed in the active environment.
2. CUDA compatibility: Ensure that your PyTorch installation is compatible with your CUDA environment.
3. Configuration paths: Check that the dataset paths and pretrained checkpoint paths in the YAML configuration files are correctly set before running experiments.
# 2. Repository Structure 📁

```text
Adaptive/
├── configs/                          # Experiment configurations
│   ├── capera/                       # CAPERA baseline configurations
│   ├── adaptive_frame_pruning.yaml   # Frame pruning configuration
│   ├── adaptive_frame_pruning_v2.yaml
│   ├── adaptive_selector.yaml        # Adaptive selector configuration
│   ├── frame_captioning.yaml         # Captioning configuration
│   └── frame_captioning_training.yaml
├── models/
│   ├── frame_cocap/                  # Frame sampling and captioning
│   │   ├── frame_sampling.py
│   │   ├── frame_video_captioner.py
│   │   ├── sensitivity_eval.py
│   │   └── training.py
│   ├── selectors/                    # Frame selection methods
│   │   ├── adaptive_frame_cocap.py
│   │   ├── adaptive_frame_pruning.py
│   │   ├── adaptive_selector.py
│   │   ├── base_selector.py
│   │   ├── caption_reward.py
│   │   ├── lightweight_frame_encoder.py
│   │   ├── picknet_style_selector.py
│   │   ├── policy_trainer.py
│   │   └── utility_critic.py
│   └── selector_enc_dec.py           # Selector encoder-decoder
├── scripts/                          # Training, inference, and evaluation
│   ├── prepare_msvd.py               # Prepare MSVD data
│   ├── train_adaptive_frame_pruning.py
│   ├── infer_adaptive_frame_pruning.py
│   ├── run_uniform8_baseline.py      # Uniform-8 baseline
│   ├── eval_capera.sh                # CAPERA evaluation
│   └── ...
└── README.md
```

Notes:

- `configs/` contains configuration files for frame pruning, selection, and captioning experiments.
- `models/frame_cocap/` contains frame sampling, captioning, and training components.
- `models/selectors/` contains frame selection strategies and related scoring and training modules.
- `scripts/` provides entry points for data preparation, training, inference, and evaluation.

# 3. Datasets 📦

The following datasets are used in our experiments.

### MSVD (Microsoft Research Video Description Corpus)

- **Official repository:** https://github.com/microsoft/Microsoft-Research-Video-Description-Corpus


MSVD contains 1,970 short video clips with multiple natural-language descriptions. We follow the standard split of 1,200 training videos, 100 validation videos, and 670 test videos.

### WebUAV-3M

- **Official GitHub repository:** https://github.com/983632847/WebUAV-3M

WebUAV-3M contains over 3 million high-definition frames from drone videos. Please refer to the official repository for dataset access instructions and download links. Due to its large size, ensure that sufficient storage is available before downloading and preparing the dataset.

# 4. Support 🛠️

If you meet issues, please open a GitHub issue with:
- Operating system and CUDA version
- Full error log
- Script and command-line arguments

If you need further assistance, please contact **Jiaren Zou** at [jiarenzou@acm.org](mailto:jiarenzou@acm.org).



























































