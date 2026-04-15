This project implements an Artificial Neural Network (ANN) using TensorFlow/Keras to predict customer churn. The model analyzes customer data and identifies whether a customer is likely to leave a service.

The project also highlights high-risk customers, enabling businesses to take proactive retention measures.

## Features
1.) Data preprocessing (encoding, scaling, transformation)  
2.) ANN model built using TensorFlow/Keras  
3.) Binary classification (Churn / Not Churn)  
4.) Model evaluation using:  
5.) Accuracy Score  
6.) Confusion Matrix  
7.) Identification of top high-risk customers  
## Technologies Used  
1.) Python    
2.) Pandas & NumPy  
3.) Scik3it-learn  
4.) TensorFlow / Keras  

## The dataset used:
Artificial_Neural_Network_Case_Study_data.csv  

## Project Workflow
### 1. Data Preprocessing  
1.) Extract independent and dependent variables  
2.) Encode categorical variables using:  
3.) Label Encoding  
4.) One-Hot Encoding  
5.) Split dataset into training and testing sets  
6.) Apply feature scaling using StandardScaler  
### 2. Model Building  
1.) The ANN architecture:  
2.) Input Layer  
3.) Hidden Layer 1: 128 neurons (ReLU)  
4.) Hidden Layer 2: 64 neurons (ReLU)  
5.) Output Layer: 1 neuron (Sigmoid)  
### 3. Model Training  
1.) Optimizer: Adam  
2.) Loss Function: Binary Crossentropy  
3.) Epochs: 50  
4.) Batch Size: 32  
### 4. Model Evaluation  
1.) Accuracy Score  
2.) Confusion Matrix  
3.) Prediction probabilities  
### 5. Business Insight  
The model identifies:  
1.) Customers with highest churn probability  
2.) Top 10 customers who need retention strategies  
3.) Sample Output  
4.) Model Accuracy  
5.) Confusion Matrix  
6.) Top 10 High-Risk Customers  
