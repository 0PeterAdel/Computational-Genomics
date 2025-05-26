# Task 1: Mutation-Based Cancer Classification

Task 1 focuses on classifying cancer patients using genomic mutation data only. This repository provides a Jupyter Notebook (`1.ipynb`) and a Python script (`1.py`) optimized for feature engineering, visualization, and machine learning classification. The code is designed to run on environments with GPU support (e.g., Google Colab) but can also operate on CPU.

The project processes mutation datasets to generate features such as total mutations, variant classifications, gene-specific mutations, mutation types (Transition, Transversion, etc.), and normalized mutation rates. Our analysis provides comprehensive visualizations of mutation patterns and their relationship to cancer subtypes.

![Genomics](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExN3IwbjduemRsaXZ4OHRwbmZyZjYyNHoxZWoxaGxhbW1mN2RjMDc4MSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/uHV4veFjX22Pu/giphy.gif)

## 📦 Project Structure

```bash
Task1/
├── 1.ipynb                  # Jupyter Notebook for Task 1 with Markdown and code cells
├── 1.py                     # Python script equivalent of 1.ipynb
├── out/                     # Contains generated outputs
│   ├── feature_importance_task1.csv  # Feature importance scores from the classifier
│   ├── mutation_distribution_by_cancer_and_variant.png  # Bar chart of mutation distribution
│   ├── mutation_distribution_by_gene.png  # Bar chart of top 20 genes by mutation count
│   └── task1_predictions.csv         # Prediction file for Task 1
└── README.md                # This file, providing instructions and details
```

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- Git

### Set Up Environment

Choose one of the following options:

#### Using Conda

```bash
# Create and activate conda environment
conda create -n task1 python=3.8
conda activate task1

# Install CUDA toolkit for GPU support
conda install -c conda-forge cudatoolkit=11.7

# Install required packages
pip install -r ../../requirements.txt
```

#### Using Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install required packages
pip install -r ../../requirements.txt
```

## 🚀 Usage

### Data Preparation

1. Ensure the following files are available:
   - `DataSet/train_muts_data.csv`
   - `DataSet/test_muts_data.csv`
   - `DataSet/train_feats.csv`
   - `DataSet/test_feats.csv`
   - `DataSet/100_genes.csv`

### Running the Code

#### Using Jupyter Notebook

```bash
jupyter notebook 1.ipynb
```

#### Using Python Script

```bash
python 1.py
```

## 📊 Outputs

The code generates the following outputs in the `out/` directory:

- `feature_importance_task1.csv`: Feature importance scores from the classifier
- `mutation_distribution_by_cancer_and_variant.png`: Distribution of mutations across cancer types
- `mutation_distribution_by_gene.png`: Top 20 genes by mutation count
- `confusion_matrix_task1.png`: Confusion matrix showing model validation performance
- `task1_predictions.csv`: Final predictions with `id_case` and `label_predict` columns

## 📈 Performance Metrics

Task 1 achieves the following baseline performance:

- F1-Score: 0.658
- Precision: 0.676
- Recall: 0.664

These metrics serve as a baseline for comparison with the integrated approach in Task 2.

---

🔬 **Happy analyzing genomic data!**

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=65&section=footer"/>
</p>