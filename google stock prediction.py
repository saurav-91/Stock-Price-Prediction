# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 23:26:59 2020

@author: SAURAV COOL
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

from datetime import date, timedelta
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

df = pd.read_csv('Google.csv')
df['Date'] = df['Date'].astype('datetime64[ns]')
df.set_index('Date', inplace = True)
df['High_Low_Change'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] *100.0
df['Change_Perc'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] *100.0

df = df[['Adj. Close','High_Low_Change','Change_Perc', 'Adj. Volume']]
df.fillna(-9999, inplace = True)

# defining labels and feature

df['Price_After_Month'] = df['Adj. Close'].shift(-30)

#Features
X = np.array(df.drop(['Price_After_Month'],1))
X = preprocessing.scale(X)
X=X[:-30]
X_Check = X[-30:]

df.dropna(inplace = True)
y = np.array(df['Price_After_Month'])

# Splitting the dataset  into training and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
accuracy = accuracy * 100
accuracy = float("{0:.2f}".format(accuracy))
print('Accuracy is :',accuracy, '%')

forecast = clf.predict(X_Check)

last_date = df.iloc[-1].name
modified_date = last_date + timedelta(days=1)
date = pd.date_range(last_date, periods = 30, freq = 'D')
df1 = pd.DataFrame(forecast, columns = ['Forecast'], index = date)
df = df.append(df1)

# to plot the predictio graph
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


















