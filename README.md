# Deep Neural Network for Customer Churn Prediction

This project demonstrates the use of a deep neural network to predict customer churn using the "Customer-Churn-Records" dataset. It addresses the class imbalance issue using SMOTE and achieves high accuracy and AUC. The model is built using PyTorch and includes data preprocessing, model training, evaluation, and saving.

## Project Overview

Customer churn, the rate at which customers stop doing business with an entity, is a critical metric for businesses. This project aims to predict customer churn using a deep neural network. The notebook provides a comprehensive workflow, from data loading and preprocessing to model training and evaluation. The key steps include:

* **Data Loading and Exploration:** The project begins by loading the dataset and performing exploratory data analysis (EDA) to understand the data distribution and relationships between features. Visualizations like count plots and kernel density estimations are used to gain insights.
* **Data Preprocessing:** Categorical features are encoded using ordinal encoding. Numerical features are standardized using `StandardScaler` to ensure they have zero mean and unit variance. Class imbalance is handled using SMOTE (Synthetic Minority Over-sampling Technique) to oversample the minority class (churned customers).
* **Model Building:** A deep neural network is constructed using PyTorch. The model consists of multiple fully connected layers with ReLU activation functions and dropout for regularization. A sigmoid activation is applied to the final layer to produce a probability for churn.
* **Model Training:** The model is trained using the Adam optimizer and binary cross-entropy loss. The training loop includes iterating over the training data in batches, computing the loss, and updating the model's weights.
* **Model Evaluation:** The trained model is evaluated on a validation set using metrics such as accuracy, precision, recall, F1-score, and AUC (Area Under the ROC Curve).
* **Model Saving:** The trained model is saved to disk for later use.

## Key Features

* **Handles Class Imbalance:**  SMOTE effectively addresses the class imbalance problem, leading to improved performance on the minority class.
* **PyTorch Implementation:** The model is implemented in PyTorch, a popular deep learning framework.
* **Comprehensive Evaluation:**  A range of evaluation metrics provides a complete picture of the model's performance.
* **Saved Model:** The `.pth` file contains the trained model weights, allowing for easy loading and reuse.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/DeepNeuralNetwork_For_Customer_Churn_Preditcion.git 
