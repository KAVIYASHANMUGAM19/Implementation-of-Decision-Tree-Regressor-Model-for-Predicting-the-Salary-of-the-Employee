# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Prepare your data -Collect and clean data on employee salaries and features
2. Split data into training and testing sets
3. Define your model -Use a Decision Tree Regressor to recursively partition data based on input features
4. Determine maximum depth of tree and other hyperparameters
5. Train your model -Fit model to training data -Calculate mean salary value for each subset
6. Evaluate your model -Use model to make predictions on testing data -Calculate metrics such as MAE and MSE to evaluate performance.
7. Tune hyperparameters -Experiment with different hyperparameters to improve performance
8. Deploy your model Use model to make predictions on new data in real-world application.
## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: S KAVIYA
RegisterNumber: 212223040090

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
# Initial dataset:
![image](https://github.com/user-attachments/assets/57840af8-43b2-4f13-b0fe-73f51ea03fed)

# Date Info:
![image](https://github.com/user-attachments/assets/005b33ef-8296-42a9-ab7e-8b0c7d6bc033)

# Optimization of null values:
![image](https://github.com/user-attachments/assets/143bbe47-49b1-43b1-86a7-8e3e465bd556)

# Converting string literals to numericl values using label encoder:
![image](https://github.com/user-attachments/assets/27fb80d1-f7ed-4a7f-a70a-1be51422c30c)

# Assigning x and y values:
![image](https://github.com/user-attachments/assets/2c3163fd-ba03-44d9-b09b-146da4063149)

# Mean Squared Error:
![image](https://github.com/user-attachments/assets/18b8d40a-702a-4197-8f0e-38633808560a)

# R2 (variance):
![image](https://github.com/user-attachments/assets/154181f2-e28e-4e1c-8cb7-b9a2d110024d)

# Prediction:
![image](https://github.com/user-attachments/assets/949bcf4f-dffb-47f1-a3a0-04574c3b1e72)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
