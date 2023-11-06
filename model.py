import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('data_cardiovascular_risk.csv')

def show_missing():
    missing = df.columns[df.isnull().any()].tolist()
    return missing
# Missing data counts and percentage
print('Missing Data Count')
print(df[show_missing()].isnull().sum().sort_values(ascending = False))
print('--'*50)
print('Missing Data Percentage')
print(round(df[show_missing()].isnull().sum().sort_values(ascending = False)/len(df)*100,2))

#Imputation
df['education'].fillna(df['education'].mode()[0],inplace=True)
df['BPMeds'].fillna(0,inplace=True)
df['cigsPerDay'].fillna(df['cigsPerDay'].mode()[0],inplace=True)
df.drop(df[(df['totChol'] == 600) | (df['totChol'] == 696)].index, inplace=True)
df['totChol'].fillna(df['totChol'].mean(),inplace=True)
df['heartRate'].fillna(df['heartRate'].mean(),inplace=True)
df['glucose'].fillna(df.glucose.median(),inplace=True)
df['BMI'].fillna(df['BMI'].mean(),inplace=True)

#Handling categorical variables
df_dummies = pd.get_dummies(df[['sex', 'is_smoking']], drop_first=True)
df.drop(['sex', 'is_smoking'],axis=1,inplace=True)
df=pd.concat([df,df_dummies],axis=1)
print("df.columns after get_dummies: ",df.columns)

#Oversampling usnIg SMOTE technique
from imblearn.over_sampling import SMOTE
smote = SMOTE()
# fit predictor and target variable
x_smote, y_smote = smote.fit_resample(df.drop(['TenYearCHD'],axis=1), df['TenYearCHD'])
print('Original dataset shape', len(df))
print('Resampled dataset shape', len(y_smote))
x_smote.drop(['id'],axis=1,inplace=True)
x_smote=x_smote.values
y_smote=y_smote.values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x_smote,y_smote,test_size=0.3,random_state=101)

from sklearn.ensemble import GradientBoostingClassifier
g = GradientBoostingClassifier(learning_rate= 0.3, max_depth= 10, n_estimators= 190)
g.fit(X_train,y_train)

pred=g.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

print(accuracy_score(y_test,pred))

import pickle
pickle.dump(g,open('cardio_model_GB.pkl','wb'))
'''model_cls = pickle.load(open('model_XGB.pkl', 'rb'))
c=[61,3,0,0,0,1,0,272,182,121,32.8,85,65,'F','NO']

li=[]
for i in c:
    i=str(i)
    if i.isalpha()==True:
        if i.lower()=='m' or i.lower()=='yes':
            li.append(1)
        else:
            li.append(0)
    else:
        li.append(float(i))
li=[li]
a=pd.DataFrame(data=li)
print(model_cls.predict(a))'''