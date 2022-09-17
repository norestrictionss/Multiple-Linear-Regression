import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import statsmodels.regression.linear_model as sm
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv("hw_tennis.csv")
play = dataset.iloc[:, -1]
others = dataset.iloc[:, :-1]
le = LabelEncoder()
logistic_regression = LogisticRegression()
outlook = pd.get_dummies(dataset['outlook'])
windy = le.fit_transform(dataset['windy'])

# Data preparation process
others = dataset.iloc[:, 1:]
new_dataset = pd.concat([outlook, others], axis=1)
new_dataset.iloc[:, -2] = le.fit_transform(dataset['windy'])
new_dataset.iloc[:, -1] = le.fit_transform(dataset['play'])
new_dataset = pd.concat([new_dataset.iloc[:, :3], new_dataset.iloc[:, 4:]], axis=1)

x = np.append(arr=np.ones((14, 1)).astype(int),
              values=new_dataset.iloc[:, :-1], axis=1)
x_l = new_dataset.iloc[:, :-1].values
r_ols = sm.OLS(endog=new_dataset.iloc[:, -1:], exog=x_l)
r = r_ols.fit()

# Regression results
print(r.summary())

# New dataset after data preparation process
print(new_dataset)

# Splitting process
x_train, x_test, y_train, y_test = train_test_split(new_dataset.iloc[:, :-1], new_dataset.iloc[:, -1], test_size=0.33)
logistic_regression.fit(x_train, y_train)
result = logistic_regression.predict(x_test)

# Comparison of prediction and test set
print(result)
print(y_test)

# Examination of accuracy by confusion matrix
print(confusion_matrix(result, y_test.values))
