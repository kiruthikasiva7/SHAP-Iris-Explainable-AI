# 🌸 SHAP Iris Explainable AI

## 📌 Project Overview

This project uses Machine Learning and Explainable AI (XAI) to classify Iris flowers.

A Random Forest Classifier is used to predict the flower species, and SHAP is used to understand why the model makes its predictions.

## 🎯 Project Objectives

- Classify Iris flowers using Machine Learning
- Train a Random Forest Classifier
- Evaluate model performance
- Use SHAP for Explainable AI
- Understand feature importance
- Test the model with custom flower measurements

## 🧠 Machine Learning Model

**Model:** Random Forest Classifier

**Dataset:** Iris Dataset

The model uses four features:

- Sepal length
- Sepal width
- Petal length
- Petal width

The model predicts one of three Iris species:

- Setosa
- Versicolor
- Virginica

## 📊 Model Performance

The model achieved **90% accuracy** on the test dataset.

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Setosa | 1.00 | 1.00 | 1.00 |
| Versicolor | 0.82 | 0.90 | 0.86 |
| Virginica | 0.89 | 0.80 | 0.84 |

## 🔍 Explainable AI with SHAP

SHAP (SHapley Additive exPlanations) is used to understand how each feature affects the model's prediction.

The project includes:

- SHAP feature importance
- Class-wise SHAP analysis
- Individual prediction explanation
- Custom flower prediction

The analysis shows that petal measurements have a strong influence on Iris classification.

## 🧪 Custom Prediction

The project allows users to enter their own flower measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

The model then predicts whether the flower is:

**Setosa, Versicolor, or Virginica.**

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- SHAP
- Matplotlib

## ▶️ How to Run

Install the required libraries:

```bash
pip install -r requirements.txt

```
## Run the project:
```bash

python shap_iris.py
```

## 📚 What I Learned
- Random Forest classification
- Model evaluation
- Explainable AI
- SHAP feature importance
- Global and local model explanations
- Custom input prediction

## 🚀 Future Improvements
- Create a Streamlit web interface
- Add interactive SHAP visualizations
- Deploy the project online

## 👩‍💻 Author

Kiruthika Siva

B.Tech Artificial Intelligence & Data Science
