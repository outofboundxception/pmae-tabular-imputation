# PMAE: Tabular Data Imputation

An implementation of **Proportionally Masked Autoencoders (PMAE)** for tabular data imputation, with **ReMasker** used as a baseline.

> **Project Title:** An Implementation of Proportionally Masked Autoencoders for Tabular Data Imputation

This project is based on:

**J. Kim, K. Lee, and T. Park, "To Predict or Not to Predict? Proportionally Masked Autoencoders for Tabular Data Imputation," AAAI-25, 2025.**

- Paper: https://ojs.aaai.org/index.php/AAAI/article/view/33967
- DOI: https://doi.org/10.1609/aaai.v39i17.33967

---

## Overview

Missing values are common in real-world tabular datasets. Simple approaches such as mean/median imputation or deleting incomplete rows may lose important relationships between features.

PMAE formulates tabular imputation as a **masked reconstruction problem**. Its main idea is to make the additional masking strategy depend on the amount of data already observed in each feature.

For feature \(j\), the observed proportion is:

\[
p_{\mathrm{obs},j}
\]

The proportional masking function used in the implementation is:

\[
M_j(p_{\mathrm{obs},j})
=
0.05
\log
\left(
\frac{p_{\mathrm{obs},j}}
{1-p_{\mathrm{obs},j}}
\right)
+0.5
\]

Therefore, different features receive different additional masking ratios according to their observed proportions.

This repository implements the PMAE workflow and compares it with a ReMasker baseline.

---

## Current Experiment

The final reported experiment uses:

| Parameter | Value |
|---|---|
| Dataset | Diabetes |
| Original samples | 442 |
| Features | 10 |
| Numerical features | 9 |
| Categorical features | 1 |
| Missingness pattern | `full` |
| Seed | `2` |
| Training device | CPU |
| Training epochs | 300 |
| PMAE architecture | MLP-Mixer |
| Baseline | ReMasker |

The current experiment is an **implementation validation on one dataset**. It is not intended to claim a complete reproduction of all experiments from the original PMAE paper.

---

## Repository Structure

The repository is intentionally kept flat so that the implementation files are directly available from the repository root.

```text
PMAE/
│
├── amputation/
│   └── diabetes/
│       ├── amputed.pkl
│       └── new_proc.pkl
│
├── pMAE.py
├── blocks.py
├── model_mae.py
├── fit_pMAE.py
├── fit_ReMasker.py
├── evaluate.py
├── configs.py
├── miss_mech.py
├── utils.py
├── requirements.txt
├── README.md
└── [supplementary] Run PMAE.ipynb
```

### Important files

| File | Purpose |
|---|---|
| `pMAE.py` | PMAE model and proportional masking implementation |
| `blocks.py` | Encoder/decoder building blocks |
| `model_mae.py` | Masked autoencoder components |
| `fit_pMAE.py` | PMAE training and imputation |
| `fit_ReMasker.py` | ReMasker baseline training and imputation |
| `evaluate.py` | Imputation evaluation metrics |
| `configs.py` | Model/configuration settings |
| `miss_mech.py` | Missingness generation utilities |
| `utils.py` | Supporting utilities |
| `[supplementary] Run PMAE.ipynb` | Main experiment notebook |

---

# Installation

## Requirements

The implementation was developed using:

- Python 3.9
- PyTorch 2.2.1
- NumPy 1.26.4
- Pandas 2.2.1
- SciPy 1.12.0
- Scikit-learn 1.4.1.post1
- einops 0.8.0
- timm 0.9.16

A GPU is **not required** for the current reported experiment because the final implementation uses CPU execution.

---

## 1. Create the Conda environment

```bash
conda create -n PMAE python=3.9 -y
```

Activate it:

```bash
conda activate PMAE
```

Check the Python version:

```bash
python --version
```

Expected:

```text
Python 3.9.x
```

---

## 2. Install dependencies

From the repository root:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Install Jupyter

The experiment is provided as a Jupyter notebook.

```bash
pip install jupyter notebook ipykernel
```

Register the environment:

```bash
python -m ipykernel install --user --name PMAE --display-name "Python 3.9 (PMAE)"
```

---

# Running the Experiment

## Using Jupyter Notebook

From the repository root:

```bash
jupyter notebook
```

Open:

```text
[supplementary] Run PMAE.ipynb
```

Select the kernel:

```text
Python 3.9 (PMAE)
```

Then run the notebook from top to bottom.

---

## Using VS Code

Install the following VS Code extensions:

- Python
- Jupyter

Open the repository folder in VS Code and open:

```text
[supplementary] Run PMAE.ipynb
```

Select:

```text
Python 3.9 (PMAE)
```

as the notebook kernel and run the cells.

---

# Current Experiment Configuration

The final notebook uses the Diabetes experiment with:

```python
dset = 'diabetes'
missing_pattern = 'full'
seed = 2
```

The prepared experiment files are located at:

```text
amputation/diabetes/
├── amputed.pkl
└── new_proc.pkl
```

These files allow the current experiment to be reproduced without regenerating the missingness configuration.

---

# PMAE Configuration

The final PMAE experiment uses:

```python
model_pmae = ProportionalMasker(args)

model_pmae.batch_size = 128
model_pmae.device = 'cpu'
model_pmae.max_epochs = 300

model_pmae.old_loss = False
model_pmae.block_mlp = 0
model_pmae.new_imp = True
```

The model configuration includes:

```text
Embedding dimension: 32
Encoder depth:       6
Decoder depth:       4
Number of heads:     4
MLP ratio:           4
Base learning rate:  0.001
Weight decay:        0.05
Warmup epochs:       40
```

### MLP-Mixer

In the implementation:

```python
block_mlp = 0
```

selects the **MLP-Mixer** architecture rather than the attention-based Transformer block.

This is consistent with the PMAE paper's investigation of MLP-Mixer based token mixing for tabular data.

---

# ReMasker Baseline

ReMasker is trained separately using:

```python
model_remasker = ReMasker(args)

model_remasker.batch_size = 128
model_remasker.device = 'cpu'
model_remasker.max_epochs = 300

model_remasker.new_imp = False
```

The same incomplete Diabetes data is used for the comparison.

---

# Implementation Pipeline

The complete workflow is:

```text
Complete Diabetes Dataset
          │
          ▼
Generate / Load Missingness
          │
          ▼
Incomplete Dataset + Missingness Mask
          │
          ├───────────────┐
          ▼               ▼
        PMAE          ReMasker
          │               │
          ▼               ▼
     Imputed Data     Imputed Data
          │               │
          └───────┬───────┘
                  ▼
              Evaluator
                  │
                  ▼
             Comparison
```

The complete ground-truth data is retained only for evaluation. The models perform imputation using the incomplete input.

---

# Evaluation Metrics

The implementation evaluates:

- **Imputation Accuracy**
- **Numerical \(R^2\)**
- **Categorical Accuracy**
- **Wasserstein Distance**
- **RMSE**
- **Numerical RMSE**
- **Categorical RMSE**

Higher is better for:

- Imputation Accuracy
- Numerical \(R^2\)
- Categorical Accuracy

Lower is better for:

- Wasserstein Distance
- RMSE

---

# Results

The final Diabetes experiment produced:

| Metric | PMAE | ReMasker |
|---|---:|---:|
| Imputation Accuracy | **0.2762** | 0.1274 |
| Numerical \(R^2\) | **0.2057** | 0.0383 |
| Categorical Accuracy | 0.8400 | 0.8400 |
| Wasserstein Distance | 0.1513 | **0.1478** |
| RMSE | **0.2037** | 0.2040 |
| Numerical RMSE | 0.1976 | **0.1973** |
| Categorical RMSE | **0.4040** | 0.4193 |

### Interpretation

For this experiment, PMAE achieves substantially better:

- Imputation Accuracy
- Numerical \(R^2\)

Both methods achieve the same categorical accuracy.

ReMasker performs slightly better on:

- Wasserstein Distance
- Numerical RMSE

Therefore, the current results demonstrate promising performance for PMAE on the Diabetes experiment, but they should **not** be interpreted as proof of universal superiority. The current evaluation uses one dataset, one missingness configuration, and one seed.

---

# Training

Both models are trained for:

```text
300 epochs
```

The final training output reported approximately:

```text
PMAE:
Loss at epoch 280 ≈ 0.16528

ReMasker:
Loss at epoch 280 ≈ 0.16938
```

The ReMasker run also encountered a non-finite loss event during training. The implementation detects invalid losses and skips the corresponding update so that training can continue.

---

# Running the Notebook from the Command Line

The notebook can also be executed without opening the Jupyter interface:

```bash
jupyter nbconvert --to notebook --execute "[supplementary] Run PMAE.ipynb" --output executed_PMAE.ipynb
```

To generate an HTML version:

```bash
jupyter nbconvert --to html executed_PMAE.ipynb
```

The resulting file will be:

```text
executed_PMAE.html
```

---

# Reproducing the Reported Experiment

To reproduce the experiment reported in the project:

1. Install Python 3.9.
2. Create the `PMAE` Conda environment.
3. Install `requirements.txt`.
4. Install Jupyter.
5. Open the repository root.
6. Open `[supplementary] Run PMAE.ipynb`.
7. Select the `Python 3.9 (PMAE)` kernel.
8. Use:
   ```python
   dset = 'diabetes'
   missing_pattern = 'full'
   seed = 2
   ```
9. Run the notebook from top to bottom.

The required prepared Diabetes files are already included under:

```text
amputation/diabetes/
```

---

# Future Work

The current implementation is intentionally evaluated on one dataset.

Future work will extend the experiment to additional tabular datasets and evaluate:

- Multiple datasets
- Multiple missingness patterns
- Multiple random seeds
- Mean and standard deviation across runs
- PMAE-MLP vs PMAE-Transformer
- ReMasker comparison
- Runtime and memory usage
- CPU vs GPU performance

The goal is to move from a single-dataset implementation study toward a broader empirical evaluation.

---

# Reference

If you use this implementation or build upon the PMAE method, please cite the original paper:

```bibtex
@inproceedings{kim2025pmae,
  title={To Predict or Not to Predict? Proportionally Masked Autoencoders for Tabular Data Imputation},
  author={Kim, Jungkyu and Lee, Kibok and Park, Taeyoung},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={17},
  pages={17886--17894},
  year={2025},
  doi={10.1609/aaai.v39i17.33967}
}
```

---

# Authors

**Bhavya Prajapati**  
Department of Computer Science  
IIIT Vadodara  
Gandhinagar, India  
202451033@iiitvadodara.ac.in

**Dhruv Garg**  
Department of Computer Science  
IIIT Vadodara  
Gandhinagar, India  
202451054@iiitvadodara.ac.in

---

## Acknowledgment

This project is an implementation and study of the PMAE method proposed by Kim, Lee, and Park. Please refer to the original paper for the complete methodology, benchmark setup, and reported experimental results.

## License

Before publishing this repository publicly, verify the licensing terms of the original PMAE implementation and any data files included in the repository. Add the appropriate license information here once confirmed.
