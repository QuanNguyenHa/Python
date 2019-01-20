import quandl, math
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'

'''
One popular option is to replace missing data with -99,999.
With many machine learning classifiers, this will just be recognized and treated as an outlier feature
'''
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
#features are a bunch of the current values, and the label shall be the price, in the future,
#where the future is 1% of the entire length of the dataset out. 
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

#drop(,1) = drop column; 0 = drop row
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

#want your features in machine learning to be in a range of -1 to 1. 
X = preprocessing.scale(X)
#print(X)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
#clf = svm.SVR(kernel='poly')
#defaul svm.SVR is linear, but can change to polynomial
#clf = svm.SVR()
#clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
