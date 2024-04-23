# EX 05 Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:

To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import Libraries: Import the necessary libraries - pandas, numpy, and matplotlib.pyplot.

2.Load Dataset: Load the dataset using pd.read_csv.

3.Remove irrelevant columns (sl_no, salary).

4.Convert categorical variables to numerical using cat.codes.

5.Separate features (X) and target variable (Y).

6.Define Sigmoid Function: Define the sigmoid function.

7.Define Loss Function: Define the loss function for logistic regression.

8.Define Gradient Descent Function: Implement the gradient descent algorithm to optimize the parameters.

9.Training Model: Initialize theta with random values, then perform gradient descent to minimize the loss and obtain the optimal parameters.

10.Define Prediction Function: Implement a function to predict the output based on the learned parameters.

11.Evaluate Accuracy: Calculate the accuracy of the model on the training data.

12.Predict placement status for a new student with given feature values (xnew).

13.Print Results: Print the predictions and the actual values (Y) for comparison.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Dhivyapriya.R
RegisterNumber: 212222230032

```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('Placement_Data.csv')
dataset

dataset=dataset.drop('sl_no',axis=1)

dataset=dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

y

theta=np.random.randn(X.shape[1])
Y=y

def sigmoid(z):
  return 1/(1+np.exp(-z))

def loss(theta,X,y):
  h=sigmoid(X.dot(theta))
  return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
  m=len(y)
  for i in range(num_iterations):
    h=sigmoid(X.dot(theta))
    gradient=X.T.dot(h-y)/m
    theta-=alpha*gradient
  return theta

theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
  h=sigmoid(X.dot(theta))
  y_pred=np.where(h>=0.5,1,0)
  return y_pred
  
y_pred=predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)

print(y_pred)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([0,0,0,0,0,2,8,2,0,0,1,0])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
## Output:

# dataset:

![image](https://github.com/dhivyapriyar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477552/6b7df026-e749-40c7-906d-3f017b853d30)

# dataset.dtypes:

![image](https://github.com/dhivyapriyar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477552/95bb2187-609c-4e3c-bace-b0ac257cf1c9)

# dataset:

![image](https://github.com/dhivyapriyar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477552/ca100799-ce4b-4f4a-9802-fa48470dbb7a)

# Y:

![image](https://github.com/dhivyapriyar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477552/97c9b014-30d1-49cd-a428-c1c5a26c91fb)

# y_pred:

![image](https://github.com/dhivyapriyar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477552/ec4c6ddd-c02a-4f67-8945-bf7b61a5e30f)

# Y:

![image](https://github.com/dhivyapriyar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477552/52013f69-7099-46c9-ba56-9494f91f3e2a)

# y_prednew:

![image](https://github.com/dhivyapriyar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477552/1031d98d-eea9-47b0-a115-0315fa9b6cf5)

# y_prednew:

![image](https://github.com/dhivyapriyar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477552/0f5e361b-c0d5-4a98-a135-fc3a3ed2164e)

## Result:

Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

