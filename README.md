# AI-Powered Student Athlete Risk Prediction System

## Project Overview

This project uses machine learning to predict whether collegiate student athletes may be at academic risk based on academic, behavioral, and schedule-related factors. The goal is to demonstrate how data science and predictive modeling can support early intervention strategies for student success.

The project was developed as part of graduate-level coursework and showcases an end-to-end machine learning workflow, including data preprocessing, feature engineering, model training, evaluation, and interpretation.

## Problem Statement

Student athletes often balance demanding practice schedules, travel, coursework, and academic expectations. Institutions may benefit from predictive tools that help identify students who may need academic support before performance issues become severe.

This project explores how machine learning can be used to identify risk patterns and support proactive academic advising.

## Technical Objectives

- Build a structured machine learning workflow for academic risk prediction
- Clean and preprocess student athlete data
- Engineer features related to academics, attendance, practice load, and motivation
- Train baseline and neural-network-style classification models
- Evaluate model performance using classification metrics
- Communicate results in a way that supports decision-making

## Tools & Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
- Matplotlib
- Logistic Regression
- Neural Network / MLP Classifier
- Confusion Matrix
- Classification Metrics

## Repository Structure

```text
student-athlete-risk-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── student_athlete_risk_prediction.ipynb
│
├── src/
│   ├── train_model.py
│   └── evaluate_model.py
│
├── data/
│   └── sample/
│       └── sample_student_athlete_data.csv
│
├── images/
│   └── results-placeholder.txt
│
├── docs/
│   └── project-summary.md
│
└── models/
    └── model-placeholder.txt
```

## Dataset Description

The sample dataset contains academic and student support-related features such as:

- study_hours_per_week
- previous_gpa
- attendance_rate
- practice_hours_per_week
- academic_motivation_score
- travel_hours_per_week
- sleep_hours_per_night
- at_risk

The target variable is `at_risk`, where:

- `0` = not currently identified as academically at risk
- `1` = identified as academically at risk

## Machine Learning Workflow

```text
Raw Student Athlete Data
        |
        v
Data Cleaning & Preprocessing
        |
        v
Feature Engineering
        |
        v
Train/Test Split
        |
        v
Model Training
        |
        v
Model Evaluation
        |
        v
Academic Risk Insights
```

## Models Used

This project includes examples of:

- Logistic Regression baseline model
- Multi-Layer Perceptron classifier
- Scikit-learn pipeline workflow

## Evaluation Metrics

Model performance can be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/QuanQuan09/student-athlete-risk-prediction.git
cd student-athlete-risk-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the training script

```bash
python src/train_model.py
```

### 5. Run the evaluation script

```bash
python src/evaluate_model.py
```

### 6. Open the notebook

```bash
jupyter notebook notebooks/student_athlete_risk_prediction.ipynb
```

## Resume Description

**AI-Powered Student Athlete Risk Prediction System**  
Developed a machine learning model using Python, Pandas, Scikit-learn, and neural network techniques to identify academic risk factors among collegiate student athletes. Performed preprocessing, feature engineering, classification modeling, and performance evaluation using confusion matrices and classification metrics.

## Ethical Considerations

This project is intended as an academic and decision-support prototype. Predictive models used in education should not replace human judgment. Any real-world use would require transparency, bias testing, privacy protections, and advisor review before intervention decisions are made.

## Future Improvements

- Add additional academic and wellness features
- Include bias and fairness analysis
- Build an interactive dashboard for advisors
- Test additional classification models
- Add model explainability with feature importance
- Deploy as a lightweight web application

## Author

LaQuandra Missick  
M.S. Information Technology  
Technical Operations Center Engineer transitioning into Data Engineering, Cloud Operations, and AI-driven Infrastructure Automation
