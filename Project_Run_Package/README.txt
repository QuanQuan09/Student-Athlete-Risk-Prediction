ECE 5494 AI Neural Network Project
Final Project Submission Package

Project Title:
Neural Network-Based Early Risk Detection for Collegiate Student-Athletes Using Educational Data Mining

Overview:
This package contains the files needed to run the final project prototype. The project trains and evaluates a baseline Logistic Regression model and an MLP neural network classifier to predict whether a student-athlete is academically at risk.

Folder Contents:
- run_project.py
  Main runnable Python script. This is the file to execute for the final project.

- requirements.txt
  Python package requirements that can be installed with pip.

- data/final_student_athlete_dataset.csv
  Prepared final dataset used by run_project.py.

- data/raw/
  Original source CSV files used during data preparation:
  * DATA (1).csv
  * dataset.csv
  * mental_resilience_dataset.csv
  * StudentPerformanceFactors.csv
  * Students Performance .csv

- notebooks/Data.ipynb
  Data preparation notebook.

- notebooks/NeuNet_Training.ipynb
  Model training/testing notebook.

- outputs/
  Output folder for model results and generated figures. The script will write fresh results here.

Installation Instructions:
1. Unzip the project package.
2. Open a terminal or Anaconda Prompt.
3. Change into the unzipped project folder:

   cd ECE5494_Final_Project_Run_Package

4. Optional but recommended: create and activate a virtual environment.

   Using conda:
   conda create -n ece5494_ai_project python=3.12
   conda activate ece5494_ai_project

   Or using Python venv:
   python -m venv .venv
   source .venv/bin/activate        # macOS/Linux
   .venv\Scripts\activate           # Windows

5. Install required packages:

   pip install -r requirements.txt

How to Run the Project:
From inside the project folder, run:

   python run_project.py

Expected Outputs:
The script will:
1. Load data/final_student_athlete_dataset.csv
2. Clean obvious leakage and ID columns from the feature matrix
3. Split the data into training and testing sets
4. Train Logistic Regression and MLP Neural Network classifiers
5. Print model metrics in the terminal
6. Save output files to the outputs/ folder

Generated files include:
- outputs/model_comparison_results.csv
- outputs/logistic_regression_classification_report.txt
- outputs/mlp_neural_network_classification_report.txt
- outputs/logistic_regression_confusion_matrix.png
- outputs/mlp_neural_network_confusion_matrix.png
- outputs/logistic_regression_roc_curve.png
- outputs/mlp_neural_network_roc_curve.png

Notes:
- All required Python packages can be installed using pip from requirements.txt.
- No external package files are required outside of pip/conda installation.
- The included dataset is far below 1 GB, so the full prepared dataset is included.
- The notebooks are included for transparency, but the grader can run the project directly using python run_project.py.

Troubleshooting:
- If Python cannot find the dataset, make sure you are running the script from the unzipped project folder and that data/final_student_athlete_dataset.csv exists.
- If package import errors occur, rerun: pip install -r requirements.txt
- If using Anaconda, ensure the correct environment is activated before running the script.
