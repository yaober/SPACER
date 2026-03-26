<p align="center">
  <img src="assets/spacer_logo.png" alt="SPACER Logo" width="260">
</p>

## 📘 SPACER: A Multi-Instance Learning Framework for Spatial Transcriptomics

**SPACER** (Spatial Analysis of Cellular Engagement and Recruitment) is a deep learning framework designed to model how immune or stromal cells *engage with* and *recruit* neighboring cells within tissue microenvironments.  
It operates directly on spatial transcriptomics data (`.h5ad`) and constructs cell-centered neighborhoods (“bags”) for multi-instance learning.

<p align="center">
  <img src="assets/model.svg" alt="SPACER model overview" width="620">
</p>

This repository provides:

- The SPACER model and training pipeline  
- Bag-construction utilities for AnnData  
- Command-line training scripts  
- Gene-level SPACER score tracking across epochs  

---

### 🧪 Training modes

`train.py` supports two modes:

- **Single mode**: standard single-dataset training or pooling all bags from all datasets as if they come from one single dataset.
- **Joint mode**: Multiple samples that are input into SPACER under the joint-sample analysis mode, with optional regularization on the *global/shared* parameters to stabilize training under heterogeneous SRT sources.

In the current MIL SPACER implementation, federated aggregation is applied to the global parameter vector:

- **Global/shared \(S\)**: `SPACER score`
- **Data specific parameters (kept local)**: `Distance`, `Gene expression` and scale paremeters

#### Single mode

```bash
python train.py \
  --training_mode single \
  --data path/to/training.csv \
  --reference_gene path/to/reference_genes.csv \
  --output_dir outputs/run1
```

#### Joint moide

Provide one CSV per SRT dataset via `--joint_data` (each CSV follows the same format as `--data`).

```bash
python train.py \
  --training_mode joint \
  --joint_data data.csv \
  --comm_rounds 50 \
  --local_epochs 1 \
  --fedprox_mu 0.01 \
  --reference_gene path/to/reference_genes.csv \
  --output_dir outputs/joint_run \
  --save_global_each_round
```

### 📄 Full Documentation  
For full installation instructions, data preparation steps, and detailed tutorials, visit:

👉 **https://spacer-readme.readthedocs.io/en/latest/**

---

### 📂 Example Data  
Example datasets used in tutorials are available here:

👉 **https://drive.google.com/drive/folders/1L1zl3Qtk31YYgdURKa5_IdPI0raQUN55?usp=sharing**

---

If you have questions or want to contribute, feel free to open an issue or pull request!
