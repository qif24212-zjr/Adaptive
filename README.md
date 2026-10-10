Temporal Context-Aware Adaptive Frame Pruning for Video Captioning
https://img.shields.io/badge/License-MIT-green.svg
https://img.shields.io/badge/Python-3.10-blue.svg
https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg

This repository provides the official implementation of our temporal context-aware adaptive frame pruning method for video captioning. The method preserves captioning quality under substantial input reduction by adaptively selecting informative frames.

Our approach models local temporal dependencies with radius r=3 and estimates frame contributions using a learnable query optimized directly with the captioning objective. A Straight-Through Estimator (STE) enables gradient propagation through the hard Top-8 frame selection process.

 Experimental Protocol
Our evaluation follows a 32-to-8 candidate-frame protocol to balance temporal coverage and computational efficiency:

Candidate Pool: 32 frames uniformly sampled at equal temporal intervals.

Retention Budget: 8 frames are dynamically retained according to their learned contributions, corresponding to a 25% retention ratio and a 75% reduction in input frames.

Optimization: During training, the frame encoder is frozen, while the scoring network and captioning model are optimized. The temporal context weight is set to λ_ctx = 1.

We evaluate on the MSVD dataset (1,970 short video clips; 1,200 train, 100 val, 670 test) and validate generalization on long-duration drone videos using the WebUAV-3M dataset. Performance is measured with BLEU-4, METEOR, ROUGE-L, and CIDEr. We compare against baselines including Uniform Sampling, Random Selection, AdaFrame, PickNet, SwinBERT, VASTA, and PEEK.

Dataset Access
The following links provide access to the datasets used in our experiments.

MSVD (Microsoft Research Video Description Corpus)

Kaggle download (recommended, most stable):
https://www.kaggle.com/datasets/mabeelabo/msvd-dataset

Original project reference (UT Austin):
https://www.cs.utexas.edu/users/ml/clamp/video/maccd.html


WebUAV-3M

Official GitHub page:
https://github.com/flycma/WebUAV-3M

The dataset contains over 3 million high-definition drone video frames and is very large. The official GitHub page provides detailed application instructions and download links via Google Drive and Baidu Netdisk.

 Project Structure
The codebase is organized into modular directories for configurations, models, and execution scripts.

configs/

capera/: YAML configuration files for different experimental setups, including adaptive_frame_pruning.yaml, adaptive_frame_pruning_v2.yaml, adaptive_selector.yaml, frame_captioning.yaml, and frame_captioning_training.yaml. It also includes baseline configurations such as picknet_style.yaml and xe_baseline.yaml.

models/

frame_cocap/: Core video captioning modules, including frame_sampling.py, frame_video_captioner.py, sensitivity_eval.py, and training.py.

selectors/: Adaptive frame selection architectures, including adaptive_selector.py, base_selector.py, picknet_style_selector.py, and selector_enc_dec.py. It also contains reinforcement learning and reward components such as caption_reward.py, policy_trainer.py, and utility_critic.py.

scripts/

Data Preparation: check_msvd.py, prepare_msvd.py, preprocess_capera.py, and predecode_capera_frames.py.

Feature Extraction: extract_capera_feats.py and extract_frame_cocap_feats.py.

Training: Training launch scripts such as train_capera.py, train_adaptive_frame_pruning.py, train_frame_cocap.py, train_capera_picknet.sh, and train_capera_xe.sh.

Inference & Evaluation: infer_adaptive_frame_pruning.py, infer_frame_cocap.py, eval_capera.sh, eval_capera_picknet.sh, and eval_captioner_progression.py.

Baselines & Analysis: run_uniform8_baseline.py, fixed_budget_sensitivity.py, and reward_sanity_check.py.

Testing: Smoke tests for various components, including smoke_test_capera.py, smoke_test_frame_cocap.py, smoke_test_picknet.py, and verify_capera_era.py.

 Environment Setup
conda create -n capera python=3.10
conda activate capera

Install PyTorch (adjust the CUDA version if necessary)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Install project dependencies
pip install -e .
pip install -r requirements.txt
