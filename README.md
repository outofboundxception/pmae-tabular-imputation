# Proportionally Masked Autoencoders for Tabular Data Imputation

An implementation and experimental evaluation of **Proportionally Masked Autoencoders (PMAE)** for missing-value imputation in tabular data.

This project is based on the AAAI 2025 paper:

> J. Kim, K. Lee, and T. Park, “To Predict or Not to Predict? Proportionally Masked Autoencoders for Tabular Data Imputation,” *Proceedings of the Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI-25)*, 2025.

**Paper:** https://ojs.aaai.org/index.php/AAAI/article/view/33967

---

## Overview

Missing values are common in real-world tabular datasets. Traditional approaches such as mean/median imputation or row deletion can lose useful relationships in the data.

PMAE treats imputation as a **masked reconstruction problem**. Its main idea is to make the additional masking strategy depend on the amount of data already observed in each feature.

For a feature \(j\), the implementation first computes its observed proportion:

\[
p_{\mathrm{obs},j}
\]

and then uses the logit-based masking function from the paper:

\[
M_j(p_{\mathrm{obs},j})
=
0.05\log\left(\frac{p_{\mathrm{obs},j}}
{1-p_{\mathrm{obs},j}}\right)+0.5
\]

The resulting masking ratio is feature-dependent rather than a single global value.

This repository also includes a **ReMasker baseline** for comparison.

---

## Project Status

### Current completed experiment

The final implementation currently evaluates:

- **Dataset:** Diabetes
- **Missingness pattern:** `full`
- **Seed:** `2`
- **PMAE architecture:** MLP-Mixer
- **Training device:** CPU
- **Training epochs:** 300
- **Baseline:** ReMasker

The current experiment is intended as an **implementation validation on one dataset**, not as a complete reproduction of every experiment in the original paper.

### Additional datasets

The repository also contains the following datasets for future/extended experiments:

1. Adult
2. Bike Sharing
3. Default of Credit Card Clients
4. Estimation of Obesity Levels Based on Eating Habits and Physical Condition
5. Letter Recognition
6. Online News Popularity
7. Online Shoppers Purchasing Intention
8. Wine Quality

These datasets are collected in the `datasets/` directory. The current final notebook is configured for the Diabetes experiment.

---

## Repository Structure

```text
PMAE-main/
│
├── datasets/
│   ├── adult/
│   ├── bike+sharing+dataset/
│   ├── default+of+credit+card+clients/
│   ├── estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition/
│   ├── letter+recognition/
│   ├── online+news+popularity/
│   ├── online+shoppers+purchasing+intention+dataset/
│   └── wine+quality/
│
└── PMAE-main/
    ├── amputation/
    │   ├── diabetes/
    │   │   ├── amputed.pkl
    │   │   └── new_proc.pkl
    │   ├── ampute_dataset.ipynb
    │   ├── miss_mech.py
    │   └── miss_mech_rem.py
    │
    ├── blocks.py
    ├── configs.py
    ├── evaluate.py
    ├── fit_pMAE.py
    ├── fit_ReMasker.py
    ├── miss_mech.py
    ├── model_mae.py
    ├── pMAE.py
    ├── utils.py
    ├── requirements.txt
    ├── README.md
    └── [supplementary] Run PMAE.ipynb
```

---

# Installation

## 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd PMAE-main
```

The repository contains a nested `PMAE-main` directory. Enter the directory containing `pMAE.py`, `fit_pMAE.py`, `configs.py`, and `requirements.txt`:

```bash
cd PMAE-main
```

You should now see files such as:

```text
pMAE.py
fit_pMAE.py
fit_ReMasker.py
configs.py
evaluate.py
requirements.txt
```

---

## 2. Create the Conda environment

The implementation was developed with **Python 3.9** and **PyTorch 2.2.1**.

```bash
conda create -n PMAE python=3.9 -y
```

Activate it:

```bash
conda activate PMAE
```

Verify:

```bash
python --version
```

Expected:

```text
Python 3.9.x
```

---

## 3. Install dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project requirements:

```bash
pip install -r requirements.txt
```

The repository pins the following core versions:

```text
torch==2.2.1
numpy==1.26.4
pandas==2.2.1
scipy==1.12.0
scikit-learn==1.4.1.post1
einops==0.8.0
timm==0.9.16
```

---

## 4. Install Jupyter

The main experiment is provided as a Jupyter notebook.

```bash
pip install jupyter notebook ipykernel
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name PMAE --display-name "Python 3.9 (PMAE)"
```

---

# Running the Current Experiment

## 1. Start Jupyter

From the inner `PMAE-main` directory:

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

## 2. Current experiment configuration

The final notebook is configured approximately as follows:

```python
basedir = './'
dset = 'diabetes'

missing_pattern = 'full'
seed = 2
```

The experiment loads the prepared Diabetes files:

```text
amputation/diabetes/new_proc.pkl
amputation/diabetes/amputed.pkl
```

These files contain the processed data and generated missingness configuration used by the current experiment.

---

# PMAE Configuration

The final experiment uses:

```python
model_pmae = ProportionalMasker(args)

model_pmae.batch_size = batch_size
model_pmae.device = 'cpu'
model_pmae.max_epochs = 300

model_pmae.old_loss = False
model_pmae.block_mlp = 0
model_pmae.new_imp = True
```

In this implementation:

- `block_mlp = 0` selects the **MLP-Mixer** architecture.
- `old_loss = False` enables the newer PMAE loss formulation.
- `new_imp = True` enables the current imputation path.
- Training is performed on CPU.
- The experiment runs for 300 epochs.

The shared model configuration uses:

```text
Embedding dimension: 32
Encoder depth:       6
Decoder depth:       4
Number of heads:     4
MLP ratio:           4
Weight decay:        0.05
Base learning rate:  1e-3
Warmup epochs:       40
```

---

# ReMasker Baseline

The same notebook also trains the ReMasker baseline:

```python
model_remasker = ReMasker(args)

model_remasker.batch_size = batch_size
model_remasker.device = 'cpu'
model_remasker.max_epochs = 300

model_remasker.new_imp = False
```

This allows PMAE and ReMasker to be evaluated on the same incomplete dataset.

---

# Evaluation

The notebook evaluates the imputed data against the retained ground-truth values.

The current evaluator reports:

- **Imputation Accuracy**
- **Numerical \(R^2\)**
- **Categorical Accuracy**
- **Wasserstein Distance**
- **RMSE**
- **Numerical RMSE**
- **Categorical RMSE**

The evaluation is performed using the missing-value mask so that the imputation performance is measured at the relevant missing positions.

---

# Current Results

The current final Diabetes experiment produced the following results:

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

In this experiment, PMAE achieved substantially higher:

- Imputation Accuracy
- Numerical \(R^2\)

Both methods achieved the same categorical accuracy.

ReMasker had slightly better:

- Wasserstein Distance
- Numerical RMSE

Therefore, the results indicate that PMAE performs strongly on this particular Diabetes experiment, but they **should not be interpreted as universal superiority** because the current experiment uses one dataset, one missingness configuration, and one seed.

---

# Training

Both models are trained for:

```text
300 epochs
```

The final notebook reports approximately:

```text
PMAE:
Loss at epoch 280 = 0.16527673998864403

ReMasker:
Loss at epoch 280 = 0.16938155073060407
```

The ReMasker run also produced a non-finite loss message during training; the implementation continued after the invalid update.

---

# Running the Notebook Non-Interactively

The notebook can also be executed from the command line.

```bash
jupyter nbconvert --to notebook --execute "[supplementary] Run PMAE.ipynb" --output executed_PMAE.ipynb
```

To export the executed notebook to HTML:

```bash
jupyter nbconvert --to html executed_PMAE.ipynb
```

This produces:

```text
executed_PMAE.html
```

---

# Using VS Code

VS Code can also be used to run the project.

Install:

- Python extension
- Jupyter extension

Open the **inner `PMAE-main` directory** in VS Code.

Open:

```text
[supplementary] Run PMAE.ipynb
```

Select:

```text
Python 3.9 (PMAE)
```

as the notebook kernel and run the cells.

---

# CPU / GPU

The final project configuration explicitly uses:

```python
model_pmae.device = 'cpu'
model_remasker.device = 'cpu'
```

Therefore, a CUDA-capable GPU is **not required for the current experiment**.

A GPU may be useful for larger datasets or extended experiments, but GPU-specific execution has not been used for the final reported experiment.

---

# Extending to Other Datasets

The repository contains additional datasets that can be used for future experiments.

The next stage of the project is to extend the pipeline to:

```text
Adult
Bike Sharing
Default of Credit Card Clients
Obesity
Letter Recognition
Online News Popularity
Online Shoppers Purchasing Intention
Wine Quality
```

For a complete experimental study, future work should evaluate:

- Multiple datasets
- Multiple missingness patterns
- Multiple random seeds
- Mean and standard deviation across runs
- PMAE-MLP vs PMAE-Transformer
- ReMasker comparison
- Runtime and memory usage
- CPU vs GPU execution

The original PMAE paper evaluates a much broader benchmark; the current repository experiment is intentionally limited to the Diabetes dataset.

---

# Reproducing the Reported Experiment

To reproduce the current reported experiment as closely as possible:

1. Create the `PMAE` Conda environment.
2. Install `requirements.txt`.
3. Install Jupyter.
4. Enter the inner `PMAE-main` directory.
5. Open `[supplementary] Run PMAE.ipynb`.
6. Select the `Python 3.9 (PMAE)` kernel.
7. Keep:
   ```python
   dset = 'diabetes'
   missing_pattern = 'full'
   seed = 2
   ```
8. Run the notebook from top to bottom.

The prepared Diabetes files required for this experiment are already included in:

```text
amputation/diabetes/
```

---

# Citation

If you use this implementation or the PMAE method in your work, please cite the original paper:

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

# Project Authors

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

This project is an implementation and study of the PMAE method proposed by Kim, Lee, and Park. Please refer to the original paper for the complete methodology, benchmark design, and reported state-of-the-art comparisons.

---

## License

This repository contains an implementation based on the referenced research work. Before publishing the repository publicly, review the licensing terms of the original PMAE implementation and each included dataset, and add the appropriate license information here.
