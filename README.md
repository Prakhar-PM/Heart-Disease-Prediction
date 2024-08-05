# Heart-Disease-Prediction
This repository contains a heart disease prediction system implemented using four different machine learning algorithms: Support Vector Machine (SVM), Decision Tree, Logistic Regression, and Random Forest. The system aims to predict whether a person has heart disease based on various medical attributes.
# Heart Disease Prediction System

## Introduction

This project aims to predict the presence of heart disease in patients using machine learning techniques. The prediction system is built using four different algorithms: Support Vector Machine (SVM), Decision Tree, Logistic Regression, and Random Forest. Each model is trained and evaluated on a dataset of patient records, with features such as age, gender, blood pressure, cholesterol levels, and other medical attributes.

## Dataset

The dataset used for this project is a heart disease dataset, which contains various medical attributes for different patients. Each record in the dataset includes:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Serum cholesterol
- Fasting blood sugar
- Resting electrocardiographic results
- Maximum heart rate achieved
- Exercise-induced angina
- ST depression induced by exercise relative to rest
- Slope of the peak exercise ST segment
- Number of major vessels (0-3) colored by fluoroscopy
- Thalassemia

The target variable indicates the presence (1) or absence (0) of heart disease.

## Models

### 1. Support Vector Machine (SVM)

The SVM model uses a linear kernel to classify the data. It is trained on the training dataset and evaluated for accuracy on both training and testing datasets.

### 2. Logistic Regression

The Logistic Regression model is used for binary classification of heart disease presence. It is trained on the training dataset and evaluated for accuracy on both training and testing datasets.

### 3. Decision Tree

The Decision Tree model builds a tree-like structure of decisions to classify the data. It is trained on the training dataset and evaluated for accuracy on both training and testing datasets.

### 4. Random Forest

The Random Forest model uses an ensemble of decision trees to improve classification accuracy. It is trained on the training dataset and evaluated for accuracy on both training and testing datasets.

## Evaluation

Each model is evaluated using accuracy metrics on both training and testing datasets. Below are the evaluation results for each model:

- **SVM**
  - Training Accuracy: 84.00%
  - Testing Accuracy: 81.67%

- **Logistic Regression**
  - Training Accuracy: 85.00%
  - Testing Accuracy: 83.33%

- **Decision Tree**
  - Training Accuracy: 100.00%
  - Testing Accuracy: 80.00%

- **Random Forest**
  - Training Accuracy: 100.00%
  - Testing Accuracy: 85.00%

## Installation

To run this project, you need to have Python installed along with the following libraries:

- numpy
- pandas
- scikit-learn
- matplotlib

You can install these dependencies using pip:

```bash
pip install numpy pandas scikit-learn matplotlib


The project also includes visualizations for data analysis:

Histograms for each variable
Bar plot for heart disease frequency across different ages


o predict whether a new patient has heart disease, you can use the trained models. Below is an example of how to make a prediction using the SVM model:

input_data = (45, 1, 2, 123, 221, 0, 0, 145, 0, 0.4, 1, 1, 3)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = svm_model.predict(input_data_reshaped)

if prediction[0] == 0:
    print("The person does not have heart disease")
else:
    print("The person has heart disease")


License
This project is licensed under the MIT License - see the LICENSE file for details.

### Explanation

- **Introduction**: Brief overview of the project and its goal.
- **Dataset**: Description of the dataset used.
- **Models**: Explanation of each machine learning model used.
- **Evaluation**: Accuracy results for each model.
- **Installation**: Instructions to install necessary dependencies.
- **Usage**: Steps to clone the repository and run the project.
- **Visualization**: Information about the data visualizations included.
- **Predicting New Data**: Example code snippet to make predictions.
- **Contributing**: Invitation for contributions.
- **License**: Licensing information.

This README provides a comprehensive overview of your project and instructions for users to replicate your work.
