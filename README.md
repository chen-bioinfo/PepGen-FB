# PepGen-FB

## Abstract

Antimicrobial peptides (AMPs) emerge as a type of promising therapeutic compounds that exhibit broad spectrum antimicrobial activity with high specificity and good tolerability. However, current AI-based AMP design strategies, which primarily rely on learning the distribution of natural AMPs, fail to overcome the inherent trade-off between antimicrobial activity and toxicity, thereby hindering their clinical translation. In this work, we propose PepGen-FB, a novel multi-objective optimization method for optimizing desired properties for AMPs iteratively. It employs a curriculum learning-guided feedback mechanism to iteratively guide the generative model to \textcolor{red}{smoothly} optimize AMPs with improving antibacterial activity and decreasing toxicity. Comprehensive experiments demonstrate that the AMPs designed by PepGen-FB substantially outperform natural prototypes in achieving an optimal balance between high antimicrobial activity and low cytotoxicity, \textcolor{red}{improved the generation success rate from 7.1\% to 96.2\% compared to the ProGen2 model. Further motif analyses provide interpretative support for the optimization process. PepGen-FB enables the seamless integration of arbitrary black-box predictors while ensuring optimization stability through curriculum-guided feedback, which establishes a novel data-driven optimization paradigm.

## Summary

This directory contains a protein/peptide sequence generation and screening pipeline built on top of ProGen2. Its main purpose is to:

- build an initial training set from `non_redundant_sequences.csv`
- score sequences with external property models, including toxicity, MIC, and perplexity
- fine-tune a ProGen2 backbone with LoRA
- iteratively generate new sequences and replace low-quality training samples with candidates that have higher MIC and lower toxicity

The full workflow is centered around `main.py`, while `generate.py`, `pred_toxin.py`, `perplexity.py`, and `attribute_util/` provide supporting functionality.

## Directory Overview

- `main.py`: main training script. Builds the initial scored dataset, runs LoRA fine-tuning, performs feedback-based replacement, and saves outputs.
- `model.py`: wraps the ProGen2 backbone and PEFT LoRA logic, and defines repetition suppression during sampling.
- `generate.py`: generates candidate sequences in batches and scores them with downstream property predictors.
- `seq_dataloader.py`: tokenizes amino acid sequences and converts them into ProGen2 input tensors.
- `pred_toxin.py`: predicts toxicity probabilities with the local `toxinpred3` model.
- `perplexity.py`: computes sequence perplexity with ProGen2 plus a LoRA checkpoint.
- `paint_physicochemical_properties.py`: analyzes and visualizes physicochemical properties of training and generated sequences.
- `data/`: example data and training data.
- `attribute_util/`: AMP/MIC property models, utility code, sequence feature processing, and inference helpers.
- `toxinpred3/`: offline toxicity prediction package and model files.

## Workflow Summary

The execution flow in `main.py` is:

1. Read the `Sequence` column from `data/non_redundant_sequences.csv`.
2. Predict toxicity scores with `pred_toxin.py`.
3. Predict MIC scores with the model stored in `attribute_util/models/mic_classifier/`.
4. Compute perplexity with `perplexity.py`.
5. Save the scored dataset to `data/train.csv`.
6. Tokenize and pad sequences with `seq_dataloader.py`.
7. Fine-tune the `../progen2-large` backbone with LoRA.
8. For each feedback round:
   - generate candidate sequences with `generate.py`
   - compute MIC, toxicity, and perplexity for candidates
   - filter candidates by threshold
   - replace low-MIC or high-toxicity samples in the training set
   - continue fine-tuning and save a new LoRA checkpoint

## Data Files

- `data/non_redundant_sequences.csv`
  - columns: `Sequence`
  - purpose: initial sequence pool for training
- `data/train.csv`
  - columns: `SEQUENCE`, `TOXIN_predict`, `MIC_predict`, `Perplexity`
  - purpose: scored training dataset
- `data/Uniprot_0.csv`
  - columns: `Name`, `Sequence`
  - purpose: reference set used mainly by the physicochemical analysis script

## Prerequisites

### 1. Backbone Model Location

The code expects the ProGen2 backbone to be available in the sibling directory:

```text
../progen2-large
```

The following files must be available there:

- `tokenizer.json`
- `configuration_progen.py`
- `modeling_progen.py`
- the corresponding Hugging Face model weights and config files

If the directory layout changes, update the `model_dir` or tokenizer paths in `model.py`, `seq_dataloader.py`, and `perplexity.py`.

### 2. Local Property Models

The repository already includes the required local model assets:

- `attribute_util/models/amp_classifier/`
- `attribute_util/models/mic_classifier/`
- `toxinpred3/model/toxinpred3.0_model.pkl`

These are loaded directly at runtime and do not need to be downloaded.

### 3. Python Environment

Python 3.9 or 3.10 is recommended. The project depends on both:

- PyTorch / Transformers / PEFT
- TensorFlow / Keras
- scikit-learn and joblib
- pandas and numpy

Using an isolated virtual environment is recommended to avoid version conflicts between PyTorch and TensorFlow.

## Installation

```bash
pip install -r requirements.txt
```

If you want to use the original `toxinpred3` standalone script in addition to the wrapper inside `pred_toxin.py`, make sure the system also provides:

- `perl`

## Typical Usage

### 1. Run the Main Pipeline

From this directory:

```bash
python main.py \
  --output_dir checkpoints/mul/ \
  --max_length 50 \
  --start_MIC_cutoff 0.70 \
  --target_MIC_cutoff 0.90 \
  --start_TOXIN_cutoff 0.55 \
  --target_TOXIN_cutoff 0.33 \
  --num_train_epochs 10 \
  --feedback_epochs 7
```

The main script will:

- generate or update `data/train.csv`
- save LoRA checkpoints under `checkpoints/.../fb_epoch_*`
- save scored candidate results to `result.csv`

### 2. Generate Candidate Sequences Only

```bash
python generate.py --output_dir checkpoints/fblora
```

Note: in its `__main__` block, `generate.py` initializes the model in "load existing LoRA checkpoint" mode, so `--output_dir` must point to an existing PEFT checkpoint.

### 3. Run Physicochemical Analysis

```bash
python paint_physicochemical_properties.py
```

This script reads `data/train.csv` and multiple experiment outputs under `checkpoints/`, then performs analysis and plotting with `modlamp`, `matplotlib`, and `seaborn`.

## Important Arguments

The main arguments in `main.py` are:

- `--output_dir`: output directory for results and LoRA checkpoints
- `--max_length`: maximum sequence length for training and generation
- `--start_MIC_cutoff`: initial MIC threshold
- `--target_MIC_cutoff`: target MIC threshold
- `--start_TOXIN_cutoff`: initial toxicity threshold
- `--target_TOXIN_cutoff`: target toxicity threshold
- `--lora_r`, `--lora_alpha`, `--lora_dropout`: LoRA hyperparameters
- `--num_train_epochs`: number of epochs per training stage
- `--feedback_epochs`: number of feedback rounds
- `--learning_rate`: learning rate

## Outputs

Common outputs include:

- `data/train.csv`: latest scored training dataset
- `checkpoints/.../fb_epoch_0`: initial LoRA fine-tuning result
- `checkpoints/.../fb_epoch_k/result.csv`: scored generated sequences for feedback round `k`
- `checkpoints/.../fb_epoch_k/train.csv`: replaced training dataset for feedback round `k`

## Notes and Caveats

1. `main.py`, `generate.py`, and `perplexity.py` hard-code GPU device IDs such as `cuda:2` and `cuda:4`. Update them if your machine uses different device indices.
2. `perplexity.py` loads `./checkpoints/mul/fb_epoch_0` by default. If this directory does not exist, perplexity computation will fail unless you change the path or prepare that checkpoint first.
3. `seq_dataloader.py` uses a fixed `DataLoader` configuration with `batch_size=64` and `num_workers=8`. You may need to reduce these values on systems with limited CPU or GPU resources.
4. `pred_toxin.py` creates temporary files named `temp_seq.aac` and `temp_seq.dpc`, then removes them automatically after prediction.
5. `paint_physicochemical_properties.py` assumes that several experiment folders already exist under `checkpoints/`, such as `mic`, `mul`, and `tox_no_curri`. The script will not run as-is if those results are missing.

## Reproducibility Checklist

For a smoother reproduction process, verify the following before running the project:

- `../progen2-large` is complete and readable
- the model files under `attribute_util/models/` are intact
- `toxinpred3/model/toxinpred3.0_model.pkl` has been extracted and is usable
- your CUDA, GPU, and PyTorch setup is compatible
- hard-coded device IDs and checkpoint paths have been adjusted for your environment

## References

- `toxinpred3/README.md`: toxicity prediction package notes
- `attribute_util/`: source code are from HydrAMP
