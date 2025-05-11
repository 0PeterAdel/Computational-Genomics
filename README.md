# Cancer Classification Project Using Genomic and Methylation Data

This project focuses on classifying cancer patients using mutation and methylation data. The code implements feature engineering, visualization, and machine learning classification for two tasks: Task 1 (using mutation data only) and Task 2 (using combined mutation and methylation data).

## Prerequisites
- **Python 3.8+**
- **Required Libraries**:  
  Install all necessary dependencies using the following command:
  ```bash
  pip install pandas numpy matplotlib scikit-learn xgboost cudf torch
  ```
- **GPU-Supported Environment**:  
  - Highly recommended for performance optimization (especially for XGBoost with `device='cuda'`).  
  - If using Google Colab, enable GPU support (Runtime > Change runtime type > GPU).

## Project Structure
The following directory structure outlines the organization of files and folders in the project:
```
Computational-Genomics/
├── DataSet/
│   ├── 100_genes.csv           # List of 100 predefined genes
│   ├── train_muts_data.csv     # Mutation data for training
│   ├── test_muts_data.csv      # Mutation data for testing
│   ├── train_feats.csv         # Feature data with labels for training
│   ├── test_feats.csv          # Feature data for testing
│   ├── train_meth_data.csv     # Methylation data for training
│   └── test_meth_data.csv      # Methylation data for testing
├── extract_sequences.py        # Script for sequence extraction (not directly used in this code)
├── Instructions/
│   ├── Challenge_2025.pdf      # Project challenge document
│   ├── Main_text_npj.pdf       # Scientific reference
│   ├── PAPER1.pdf              # Additional scientific reference
│   └── תרגול 4 - אתגר הקורס.pdf  # Educational document
├── main.ipynb                  # Main project code (Jupyter Notebook)
└── output/
    ├── mutation_distribution_by_variant.png  # Bar chart of mutation distribution by variant type
    ├── mutation_distribution_by_gene.png     # Bar chart of mutation distribution by top 20 genes
    ├── task1_predictions.csv                 # Predictions for Task 1
    └── task2_predictions.csv                 # Predictions for Task 2
```

## Installation and Usage

### 1. Install Dependencies
Ensure all required libraries are installed by running:
```bash
pip install pandas numpy matplotlib scikit-learn xgboost cudf torch
```

### 2. Organize Data
- Place all CSV data files in the `DataSet/` directory as shown in the project structure.
- If using Google Colab, upload the `DataSet/` folder to `/content/work`:
  - Use the following Python command to change the directory:
    ```python
    import os
    os.chdir('/content/work')
    ```

### 3. Run the Code
- **Using Jupyter Notebook**:
  - Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
  - Open and run all cells in `main.ipynb`.
  - If using Google Colab, upload `main.ipynb` and run it directly.
- **Using Python Script (Optional)**:
  - Convert the notebook to a Python script:
    ```bash
    jupyter nbconvert --to script main.ipynb
    ```
  - Run the script:
    ```bash
    python main.py
    ```

### 4. Outputs
- **Predictions**:
  - `output/task1_predictions.csv`: Contains predictions for Task 1 (mutation-based classification).
  - `output/task2_predictions.csv`: Contains predictions for Task 2 (combined mutation and methylation classification).
- **Visualizations**:
  - `output/mutation_distribution_by_variant.png`: Bar chart showing the distribution of mutations by variant type.
  - `output/mutation_distribution_by_gene.png`: Bar chart showing the mutation distribution for the top 20 genes.

## Important Notes
- **GPU Enablement**:
  - Ensure GPU is enabled in Google Colab for optimal performance (Runtime > Change runtime type > GPU).
  - The code checks for GPU availability using `torch.cuda.is_available()` and will terminate if GPU is not detected.
- **Input Data Requirements**:
  - Ensure all input CSV files contain the required columns (e.g., `case_id`, `Gene_name`, `beta_val` for methylation files, and `Variant_Classification` for mutation files).
- **Troubleshooting**:
  - If you encounter errors, verify that all libraries are correctly installed and that the data files are properly uploaded.
- **File Path Adjustments**:
  - If running outside Colab, adjust the `os.chdir()` path to match your local directory structure (e.g., `os.chdir('DataSet/')`).
