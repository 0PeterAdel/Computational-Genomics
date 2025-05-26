# Task 2: Integrated Mutation and Methylation Cancer Classification

Task 2 extends Task 1 by integrating both genomic mutation and DNA methylation data for improved cancer classification. This repository provides a Jupyter Notebook (`2.ipynb`) and a Python script (`2.py`) that combines mutation features with methylation data to create a more comprehensive classifier. The code is optimized for GPU environments (e.g., Google Colab) but can also run on CPU.

The project processes both mutation and methylation datasets to generate integrated features such as methylation averages, standard deviations, hypermethylation proportions, and mutation-methylation interactions. Our results show this integrated approach achieves significantly better classification performance than using mutation data alone.

![Genomics](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExN3IwbjduemRsaXZ4OHRwbmZyZjYyNHoxZWoxaGxhbW1mN2RjMDc4MSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/uHV4veFjX22Pu/giphy.gif)

## 📦 Project Structure

```bash
Task2/
├── 2.ipynb                  # Jupyter Notebook for Task 2 with Markdown and code cells
├── 2.py                     # Python script equivalent of 2.ipynb
├── out/                     # Contains generated outputs
│   ├── train_combined.csv   # Combined mutation and methylation features for training
│   ├── test_combined.csv    # Combined mutation and methylation features for testing
│   ├── confusion_matrix_task2.png   # Confusion matrix visualization of model performance
│   └── task2_predictions.csv # Final predictions for Task 2
└── README.md               # This file, providing instructions and details
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
conda create -n task2 python=3.8
conda activate task2

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
   - `DataSet/train_meth_data.csv`
   - `DataSet/test_meth_data.csv`
   - `DataSet/train_feats.csv`
   - `DataSet/test_feats.csv`
   - `DataSet/100_genes.csv`

### Running the Code

#### Using Jupyter Notebook

```bash
jupyter notebook 2.ipynb
```

#### Using Python Script

```bash
python 2.py
```

## 📊 Outputs

The code generates the following outputs in the `out/` directory:

- `train_combined.csv`: Combined mutation and methylation features for training
- `test_combined.csv`: Combined mutation and methylation features for testing
- `task2_predictions.csv`: Final predictions with `id_case` and `label_predict` columns

## 📈 Performance Improvements

Task 2 achieves significant improvements over Task 1:

- F1-Score: 0.882 (vs 0.658 in Task 1)
- Precision: 0.882 (vs 0.676 in Task 1)
- Recall: 0.881 (vs 0.664 in Task 1)

These improvements demonstrate the value of integrating methylation data with mutation data for cancer classification.

---

🔬 **Happy analyzing genomic data!**
```

## 🚀 Installation

### 1. Clone the Repository

```bash
cd Computational-Genomics
```

### 2. Set Up the Environment

#### Option 1: Using Conda
Create a new Conda environment with Python 3.8:

```bash
conda create -n genomics python=3.8
conda activate genomics
```

#### Option 2: Using Python Virtual Environment
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

### 3. Install Required Dependencies

Install the necessary dependencies (including `cudatoolkit` for GPU support if using Conda):

#### If Using Conda:
```bash
conda install -c conda-forge cudatoolkit=11.7
pip install -r requirements.txt
```

#### If Using Virtual Environment:
```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes libraries such as `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`, `cudf`, and `torch`.

## Usage

After setting up the environment and ensuring the datasets are in the `DataSet/` folder, you can run the project using either the Jupyter Notebook or the Python script.

### Option 1: Using Jupyter Notebook
- **On Windows**:
  ```bash
  jupyter notebook
  ```
- **On Linux/macOS**:
  ```bash
  jupyter notebook
  ```
Open `main.ipynb` in the Jupyter interface and run all cells.

### Option 2: Using Python Script
- **On Windows**:
  ```bash
  python main.py
  ```
- **On Linux/macOS**:
  ```bash
  python main.py
  ```

Make sure to adjust the working directory in the script (`os.chdir('/content/work')`) if running outside Google Colab (e.g., `os.chdir('DataSet/')` for local execution).

---

🤍 Thank you for checking out **Computational Genomics**! Happy analyzing!


<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=65&section=footer"/>
</p>
