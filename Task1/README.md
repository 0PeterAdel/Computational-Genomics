# Task 1: Mutation-Based Cancer Classification

Task 1 focuses on classifying cancer patients using genomic mutation data only. This repository provides a Jupyter Notebook (`1.ipynb`) and a Python script (`1.py`) optimized for feature engineering, visualization, and machine learning classification. The code is designed to run on environments with GPU support (e.g., Google Colab) but can also operate on CPU.

The project processes mutation datasets to generate features such as total mutations, variant classifications, gene-specific mutations, mutation types (Transition, Transversion, etc.), strand bias, and normalized mutation rates. It includes visualizations (e.g., mutation distribution by cancer type) and produces prediction files for analysis.

![Genomics](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExN3IwbjduemRsaXZ4OHRwbmZyZjYyNHoxZWoxaGxhbW1mN2RjMDc4MSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/uHV4veFjX22Pu/giphy.gif)

## 📦 Project Structure

```
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

## 🚀 Installation

### 1. Set Up the Environment

#### Option 1: Using Google Colab
- Open `1.ipynb` in Google Colab.
- Ensure GPU runtime is enabled (Runtime > Change runtime type > GPU).

#### Option 2: Local Environment with Conda
Create a new Conda environment with Python 3.8:

```bash
conda create -n task1 python=3.8
conda activate task1
```

#### Option 3: Local Environment with Virtual Environment
Create a virtual environment using `venv`:

```bash
python -m venv .venv
```

Activate the virtual environment:

- **On Windows**:
  ```bash
  .venv\Scripts\activate
  ```
- **On Linux/macOS**:
  ```bash
  source .venv/bin/activate
  ```

### 2. Install Required Dependencies

Install the necessary libraries (including GPU support if using Conda):

#### If Using Conda:
```bash
conda install -c conda-forge cudatoolkit=11.7
pip install -r ../../requirements.txt
```

#### If Using Virtual Environment:
```bash
pip install -r ../../requirements.txt
```

The `requirements.txt` file (in the root directory) includes `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`, `cudf`, and `torch`.

## Usage

Ensure the raw datasets (`train_muts_data.csv`, `test_muts_data.csv`, `train_feats.csv`, `test_feats.csv`, `100_genes.csv`) are uploaded to the `/content/work` directory in Google Colab or the local `Task1/` directory.

### Option 1: Using Jupyter Notebook
- Open `1.ipynb` in Jupyter Notebook or Google Colab.
- Run all cells in sequence to generate features, visualizations, and predictions.

### Option 2: Using Python Script
- Run the script from the command line:
  - **On Windows**:
    ```bash
    python 1.py
    ```
  - **On Linux/macOS**:
    ```bash
    python 1.py
    ```
- Adjust the working directory (`os.chdir('/content/work')`) to match your local path (e.g., `os.chdir('Task1/')`) if not using Colab.

## Outputs
- `out/feature_importance_task1.csv`: Table of feature importance scores.
- `out/mutation_distribution_by_cancer_and_variant.png`: Visualization of mutation distribution by cancer type.
- `out/mutation_distribution_by_gene.png`: Visualization of top 20 genes by mutation count.
- `out/task1_predictions.csv`: Prediction file with `id_case` and `label_predict` columns.

---

🤍 Thank you for exploring **Task 1**! Happy analyzing!

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=65&section=footer"/>
</p>