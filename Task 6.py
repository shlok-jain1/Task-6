#!/usr/bin/env python
# coding: utf-8

# In[93]:


import pandas as pd
data1 = pd.read_excel("C:\\Users\\Admin\\Desktop\\Task 6 dataset\\1st task.xlsx")
print(data1.head())

import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(data1["date"],data1["sales"],marker="o", linestyle="-", label="Sales")
plt.title("Time series for sales data")
plt.xlabel("Date", rotation = 45)
plt.ylabel("Sales")
plt.grid(True)
plt.show()
#there seems to be annual variation in the data, for example, sales shoot up during the year end and slow down during April

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.dates as mdates

train = data1.iloc[:-12]
test = data1.iloc[-12:]
model = ARIMA(train["sales"], order=(1,1,1))
fit = model.fit()
forecast = fit.forecast(steps=len(test))

rmse = np.sqrt(mean_squared_error(test["sales"], forecast))
mape = mean_absolute_percentage_error(test["sales"], forecast)
print("\nRoot mean square error:",rmse,"\nMean Absolute percentage error:", mape)

plt.figure(figsize=(12,5))
plt.plot(train["date"],train["sales"], label="Training data", color="green")
plt.plot(test["date"],test["sales"],label="Test data", color="red")
plt.plot(test["date"], forecast, label="Forecasted Sales", color="blue", linestyle="--")
plt.title("Forecasted Sales using ARIMA(1,1,1)")
plt.ylabel("Sales")
plt.xlabel("Date")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#task2 
data2 = pd.read_excel("C:\\Users\\Admin\\Desktop\\Task 6 dataset\\2nd task.xlsx")
print(data2.head())
print(data2.isnull().sum())
data2.duplicated().sum()
data2.drop_duplicates(inplace=True)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

data2[["age","restbp","chol"]] = scaler.fit_transform(data2[["age","restbp","chol"]])
print(data2.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

X= data2[["age","restbp","chol","sex"]]
y= data2["target"]

X.loc[:, "sex"]= X["sex"].map({"male":0,"female":1})

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.8, random_state=1)

model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

conf_mat =  confusion_matrix(y_test,y_pred)
class_report = classification_report(y_test,y_pred)

print("\nConfusion Matrix:\n", conf_mat)
p = precision_score(y_test, y_pred)
r = recall_score(y_test, y_pred)
f = f1_score(y_test,y_pred)
print(f"\nPrecision: {p:.2f}")
print(f"\nRecall: {r:.2f}")
print(f"\nF1 Score: {f:.2f}")


# In[ ]:




