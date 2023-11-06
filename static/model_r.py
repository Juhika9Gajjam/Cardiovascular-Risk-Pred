import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

df=pd.read_csv('over_ml_rainfall_data.csv')
pd.set_option('display.max_columns',None)
df.head()

X=df.drop(['rainfall'],axis=1)
y=df['rainfall']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
xgb_model = xgb.XGBRegressor()

grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 500, 1000],
    'subsample': [0.5, 0.8, 1.0]
}

rg=GridSearchCV(xgb_model, param_grid = grid, scoring = 'neg_mean_absolute_error', cv=5)

rg.fit(X_train,y_train)

print(rg.best_estimator_)

pre=rg.predict(X_test)

print(mean_absolute_error(y_test,pre))
print(mean_squared_error(y_test,pre))
print(mean_squared_error(y_test,pre)**0.5)
print(r2_score(y_test,pre))

import pickle
pickle.dump(rg,open('model_xgb_3sets.pkl','wb') )
    
clf_xgb = pickle.load(open('model_xgb_3sets.pkl', 'rb'))

c=[[1982,4,95.78,26.44,34.19,7.73,267.56,7.76]]
c=ss.transform(c)

print(clf_xgb.predict(c)) ##correct one 14.2322