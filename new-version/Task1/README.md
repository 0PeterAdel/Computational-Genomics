# 🧬 Task 1: Mutation-Based Cancer Classification

Task 1 focuses on **classifying cancer patients using genomic mutation data only**.  
This repository provides a Jupyter Notebook (`1.ipynb`) and a Python script (`1.py`) optimized for:

- 🔍 Feature engineering  
- 📊 Visualization  
- 🤖 Machine learning classification  

The code is designed to run on environments with **GPU support** (e.g., Google Colab), but it also works on CPU.

---

## 🧪 Project Description

The project processes mutation datasets to generate features such as:

- Total mutations
- Variant classifications
- Gene-specific mutations
- Mutation types: *Transition*, *Transversion*, etc.
- Normalized mutation rates

The analysis includes **comprehensive visualizations** of mutation patterns and their relationship to cancer subtypes.

> **Note**: Feature extraction uses `extract_sequences.py` to retrieve *flanking sequences* around mutation sites.  
This improves mutation pattern analysis by capturing the **local genomic context**.

![Genomics](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExN3IwbjduemRsaXZ4OHRwbmZyZjYyNHoxZWoxaGxhbW1mN2RjMDc4MSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/uHV4veFjX22Pu/giphy.gif)

---

## 📦 Project Structure

```

Task1/
├── 1.ipynb                            # Jupyter Notebook for Task 1
├── 1.py                               # Python script version of 1.ipynb
├── out/                               # Generated outputs
│   ├── feature\_importance\_task1.csv   # Feature importance scores
│   ├── mutation\_distribution\_by\_cancer\_and\_variant.png
│   ├── mutation\_distribution\_by\_gene.png
│   ├── confusion\_matrix\_task1.png     # Validation performance
│   └── task1\_predictions.csv          # Final predictions
└── README.md                          # Project instructions and documentation

````

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher  
- CUDA-capable GPU (recommended) or CPU  
- Git  

### 🛠️ Set Up Environment

#### Option 1: Using Conda
```bash
# Create and activate conda environment
conda create -n task1 python=3.8
conda activate task1

# Install CUDA toolkit
conda install -c conda-forge cudatoolkit=11.7

# Install required packages
pip install -r requirements.txt
````

#### Option 2: Using Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

---

## 🚀 Usage

### 📁 Data Preparation

Ensure the following files are available in `DataSet/`:

* `train_muts_data.csv`
* `test_muts_data.csv`
* `train_feats.csv`
* `test_feats.csv`
* `100_genes.csv`

### ▶️ Running the Code

#### Using Jupyter Notebook:

```bash
jupyter notebook 1.ipynb
```

#### Using Python Script:

```bash
python 1.py
```

---

## 📊 Outputs

The following outputs are saved in the `out/` directory:

* `feature_importance_task1.csv`: Feature importance scores
* `mutation_distribution_by_cancer_and_variant.png`: Mutation distribution across cancer types
* `mutation_distribution_by_gene.png`: Top 20 mutated genes
* `confusion_matrix_task1.png`: Validation performance
* `task1_predictions.csv`: Final predictions with `id_case` and `label_predict` columns

> **Note**: Visualizations use **combined train/test data** for a complete overview.

---

## 📈 Performance Metrics

| Metric    | Value |
| --------- | ----- |
| F1-Score  | 0.658 |
| Precision | 0.676 |
| Recall    | 0.664 |

> These metrics are calculated using an **80/20 validation split** from training data.
> They serve as the baseline for comparison with Task 2.

---

## 🧬 Additional Notes

* **Strand Bias Feature**: Removed due to constant value (noise reduction).
* **Mutation Type Classification**: `classify_mutation()` categorizes mutations into:

  * Transition
  * Transversion
  * Deletion
  * Insertion
  * Other
* **Classification Models**:

  * RandomForest
  * XGBoost
  * Hyperparameter tuning via `GridSearchCV`
  * Best model selected based on **F1-Score**

---

## 🔬 **Happy analyzing genomic data!**

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=65&section=footer"/>
</p>
