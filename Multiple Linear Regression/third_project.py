import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import statsmodels.regression.linear_model as sm

dataset = pd.read_csv("hw_tennis.csv")
play_issue = dataset.iloc[:, -1]
others = dataset.iloc[:, :-1]
le = LabelEncoder()
linear_regression = LinearRegression()
outlook = pd.get_dummies(dataset['outlook'])
windy = le.fit_transform(dataset['windy'])

others = dataset.iloc[:, 1:]
new_dataset = pd.concat([outlook, others], axis=1)
new_dataset.iloc[:, -2] = le.fit_transform(dataset['windy'])
new_dataset.iloc[:, -1] = le.fit_transform(dataset['play'])
new_dataset = pd.concat([new_dataset.iloc[:, :3], new_dataset.iloc[:, 4:]],axis=1)

x = np.append(arr=np.ones((14, 1)).astype(int),
              values=new_dataset.iloc[:, :-1], axis=1)
x_l = new_dataset.iloc[:, :-1].values
r_ols = sm.OLS(endog=new_dataset.iloc[:, -1:], exog=x_l)
r = r_ols.fit()

print(r.summary())
print(new_dataset)
x_train, x_test, y_train, y_test = train_test_split(new_dataset.iloc[:, :-1], new_dataset.iloc[:, -1], test_size=0.33)
linear_regression.fit(x_train, y_train)
result = linear_regression.predict(x_test)
print(result)
print(y_test)
