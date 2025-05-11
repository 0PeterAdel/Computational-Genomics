# Computational Genomics

Computational Genomics is a project focused on classifying cancer patients using genomic mutation and methylation data. This repository provides scripts for feature engineering, visualization, and machine learning classification, enabling two tasks: Task 1 (classification using mutation data only) and Task 2 (classification using combined mutation and methylation data). The project is optimized for environments with GPU support but can also run on CPU, making it suitable for a variety of hardware configurations.

The training data consists of mutation and methylation datasets, which are processed into features for machine learning models. The raw datasets are available in the `DataSet/` folder, and the project includes scripts to generate visualizations and predictions for analysis.

![Genomics](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExN3IwbjduemRsaXZ4OHRwbmZyZjYyNHoxZWoxaGxhbW1mN2RjMDc4MSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/uHV4veFjX22Pu/giphy.gif)

## 📦 Project Structure

```
Computational-Genomics/
├── DataSet/                    # Contains the raw datasets for training and testing
│   ├── 100_genes.csv           # List of 100 predefined genes
│   ├── train_muts_data.csv     # Mutation data for training
│   ├── test_muts_data.csv      # Mutation data for testing
│   ├── train_feats.csv         # Feature data with labels for training
│   ├── test_feats.csv          # Feature data for testing
│   ├── train_meth_data.csv     # Methylation data for training
│   └── test_meth_data.csv      # Methylation data for testing
├── extract_sequences.py        # Script for sequence extraction (not directly used in this code)
├── Instructions/               # Contains project documentation and references
│   ├── Challenge_2025.pdf      # Project challenge document
│   ├── Main_text_npj.pdf       # Scientific reference
│   ├── PAPER1.pdf              # Additional scientific reference
│   └── pptx.pdf                # Presentation or additional instructional material
├── main.ipynb                  # Main project code (Jupyter Notebook)
├── main.py                     # Main project code (Python script)
├── output/                     # Contains generated outputs (predictions and visualizations)
│   ├── mutation_distribution_by_variant.png  # Bar chart of mutation distribution by variant type
│   ├── mutation_distribution_by_gene.png     # Bar chart of mutation distribution by top 20 genes
│   ├── task1_predictions.csv                 # Predictions for Task 1
│   └── task2_predictions.csv                 # Predictions for Task 2
├── README.md                   # This file, providing project instructions and details
└── requirements.txt            # File listing all required Python libraries
```

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/0PeterAdel/Computational-Genomics.git
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