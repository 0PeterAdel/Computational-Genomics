
# Task 2: Mutation and Methylation-Based Cancer Classification

Task 2 extends the classification to use both genomic mutation and methylation data. This repository provides a Jupyter Notebook (`2.ipynb`) and a Python script (`2.py`) optimized for feature engineering and machine learning classification. The code is designed for GPU-supported environments (e.g., Google Colab) but can also run on CPU.

The project processes mutation and methylation datasets to create combined features, including mutation counts, variant classifications, methylation averages, standard deviations, high-methylation proportions, and interaction features. It generates prediction files for analysis.

![Genomics](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExN3IwbjduemRsaXZ4OHRwbmZyZjYyNHoxZWoxaGxhbW1mN2RjMDc4MSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/uHV4veFjX22Pu/giphy.gif)

## 📦 Project Structure

```
Task2/
├── 2.ipynb                  # Jupyter Notebook for Task 2 with Markdown and code cells
├── 2.py                     # Python script equivalent of 2.ipynb
├── out/                     # Contains generated outputs
│   ├── task2_predictions.csv  # Prediction file for Task 2
│   ├── test_combined.csv      # Combined test data (saved for reference)
│   └── train_combined.csv     # Combined training data (saved for reference)
└── README.md                # This file, providing instructions and details
```

## 🚀 Installation

### 1. Set Up the Environment

#### Option 1: Using Google Colab
- Open `2.ipynb` in Google Colab.
- Ensure GPU runtime is enabled (Runtime > Change runtime type > GPU).

#### Option 2: Local Environment with Conda
Create a new Conda environment with Python 3.8:

```bash
conda create -n task2 python=3.8
conda activate task2
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

The `requirements.txt` file (in the root directory) includes `pandas`, `numpy`, `scikit-learn`, `xgboost`, `cudf`, and `torch`.

## Usage

Ensure the raw datasets (`train_muts_data.csv`, `test_muts_data.csv`, `train_feats.csv`, `test_feats.csv`, `train_meth_data.csv`, `test_meth_data.csv`, `100_genes.csv`) are uploaded to the `/content/work` directory in Google Colab or the local `Task2/` directory.

### Option 1: Using Jupyter Notebook
- Open `2.ipynb` in Jupyter Notebook or Google Colab.
- Run all cells in sequence to generate combined features and predictions.

### Option 2: Using Python Script
- Run the script from the command line:
  - **On Windows**:
    ```bash
    python 2.py
    ```
  - **On Linux/macOS**:
    ```bash
    python 2.py
    ```
- Adjust the working directory (`os.chdir('/content/work')`) to match your local path (e.g., `os.chdir('Task2/')`) if not using Colab.

## Outputs
- `out/task2_predictions.csv`: Prediction file with `id_case` and `label_predict` columns.
- `out/test_combined.csv`: Saved combined test data for reference.
- `out/train_combined.csv`: Saved combined training data for reference.

---

🤍 Thank you for exploring **Task 2**! Happy analyzing!

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=65&section=footer"/>
</p>