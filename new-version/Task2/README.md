# 🧬 Task 2: Integrated Mutation and Methylation Cancer Classification

Task 2 extends Task 1 by integrating genomic **mutation** and **DNA methylation** data for enhanced cancer classification. This repository provides a Jupyter Notebook (`2.ipynb`) and a Python script (`2.py`) that combine mutation features with methylation data to create a robust classifier. The code is optimized for **GPU environments** (e.g., Google Colab) but can also run on CPU.

The project processes mutation and methylation datasets to generate integrated features such as:

* Methylation averages and standard deviations
* Hypermethylation proportions
* Mutation-methylation interactions

Visualizations, including a **confusion matrix**, are provided to illustrate model performance. The integrated approach achieves significantly better classification performance than mutation data alone.

---
![Genomics](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExN3IwbjduemRsaXZ4OHRwbmZyZjYyNHoxZWoxaGxhbW1mN2RjMDc4MSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/uHV4veFjX22Pu/giphy.gif)

## 📦 Project Structure

```bash
Task2/
├── 2.ipynb                        # Jupyter Notebook with code & markdown
├── 2.py                           # Python script version
├── out/                           # Output directory
│   ├── confusion_matrix_task2.png     # Confusion matrix
│   ├── task2_predictions.csv          # Final predictions
│   ├── test_combined.csv             # Combined test features
│   └── train_combined.csv            # Combined training features
└── README.md                     # Instructions and project overview
```

---

## 🔧 Installation

### Prerequisites

* Python 3.8 or higher
* CUDA-capable GPU (recommended) or CPU
* Git

### Set Up Environment

#### ✅ Using Conda

```bash
# Create and activate environment
conda create -n task2 python=3.8
conda activate task2

# Install CUDA toolkit
conda install -c conda-forge cudatoolkit=11.7

# Install required packages
pip install -r requirements.txt
```

#### ✅ Using Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate

# Install required packages
pip install -r /requirements.txt
```

---

## 🚀 Usage

### 📁 Data Preparation

Ensure the following files are available inside the `DataSet/` directory:

* `DataSet/train_muts_data.csv`
* `DataSet/test_muts_data.csv`
* `DataSet/train_meth_data.csv`
* `DataSet/test_meth_data.csv`
* `DataSet/train_feats.csv`
* `DataSet/test_feats.csv`
* `DataSet/test_mutation_data.csv`
* `DataSet/100_genes.csv`

---

### ▶️ Running the Code

#### Using Jupyter Notebook

```bash
jupyter notebook 2.ipynb
```

#### Using Python Script

```bash
python 2.py
```

> ⚠️ **Note**: Adjust the working directory in the script if you're not using Google Colab:
> e.g., change `os.chdir('/content/work')` to `os.chdir('DataSet/')`.

---

## 📊 Outputs

The following outputs are generated in the `out/` directory:

* `confusion_matrix_task2.png`: Confusion matrix showing model validation performance
* `task2_predictions.csv`: Final predictions (`id_case`, `label_predict`)
* `test_combined.csv`: Combined test features
* `train_combined.csv`: Combined training features

---

## 📈 Performance Metrics

Task 2 achieves **significant improvements** over Task 1:

| Metric        | Task 2 | Task 1 |
| ------------- | ------ | ------ |
| **F1-Score**  | 0.882  | 0.658  |
| **Precision** | 0.882  | 0.676  |
| **Recall**    | 0.881  | 0.664  |

* Metrics are based on a **20% stratified validation split**.
* See `confusion_matrix_task2.png` for classification visualization (HNSC vs. LUSC).

---

## 🔬 Additional Notes

* **Strand Bias Feature**: Removed as it was constant and added no value to the model.
* **Mutation Type Classification**: The `classify_mutation` function categorizes mutations into:

  * Transition
  * Transversion
  * Deletion
  * Insertion
  * Other
* **Model Selection**: Models used:

  * `RandomForestClassifier`
  * `XGBClassifier` with GPU acceleration
  * Hyperparameter tuning via `GridSearchCV`
* **Validation**: Based on 20% stratified split, weighted F1-score used for model comparison.
* **Visualization**: Confusion matrix created for validation results.

---

## 📥 Clone the Repository

```bash
cd Computational-Genomics/Task2
```

---

🔬 **Happy analyzing genomic data!**

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=65&section=footer"/>
</p>
