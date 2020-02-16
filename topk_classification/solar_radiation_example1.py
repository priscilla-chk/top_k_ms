#!/usr/bin/env python
# coding: utf-8


from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
import missingno as ms

##Import topk classification module
from topk_model_selector import *

##For machine learning purposes
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import metrics


solardata = pd.read_csv('SolarPrediction.csv', parse_dates=['Data'])
solardata['Date_s'] = pd.to_timedelta(solardata['Data']).dt.total_seconds().astype('float')
solardata['Time_s'] = pd.to_timedelta(solardata['Time']).dt.total_seconds().astype('float')
solardata['TimeSunRise_s'] = pd.to_timedelta(solardata['TimeSunRise']).dt.total_seconds().astype('float')
solardata['TimeSunSet_s'] = pd.to_timedelta(solardata['TimeSunSet']).dt.total_seconds().astype('float')
solardata = solardata.drop(['UNIXTime','Data','Time','TimeSunRise','TimeSunSet'],axis=1)
print(solardata.head(5))

#ms.matrix(solardata)
print(solardata.info())
print(solardata.shape)
print(solardata.describe())

ax1=pd.plotting.scatter_matrix(solardata, figsize=(8,8))
n = len(solardata.select_dtypes(include=['number']).columns)
for x in range(n):
    for y in range(n):
        ax = ax1[x, y] 
        ax.xaxis.label.set_rotation(90)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.labelpad = 50
plt.show()

solardata['solar_radiation'] = pd.cut(solardata['Radiation'], bins=4, labels = None)
solardata['solar_radiation'].value_counts(dropna=False)

x = solardata.drop(['Radiation','solar_radiation'], axis=1)
y = solardata['solar_radiation']

#save list of indices before splitting
yinds = arange(y.shape[0]).tolist()

x_train, x_test, y_train, y_test, yinds_train, yinds_test = train_test_split(x,y,yinds,test_size=0.25,random_state=10)
print('Training set size: ', x_train.shape, y_train.shape)
print('Test set size:', x_test.shape, y_test.shape)

scaler = StandardScaler()
scaler.fit(x_train)
x_train_trans = scaler.transform(x_train)
x_test_trans = scaler.transform(x_test)

yencoder = LabelEncoder()
yencoder.fit(y_train)
y_train_trans = yencoder.transform(y_train)
y_test_trans = yencoder.transform(y_test)

print(x_train_trans.shape, x_test_trans.shape)

##No. of matches 
n_pred = 2

m1 = KNeighborsClassifier(n_neighbors=7)
m2 = GaussianNB()
m3 = LogisticRegression(random_state=0,solver='newton-cg',multi_class='multinomial')
k_models = [m1,m2,m3]

##Calculate Top-k accuracy using K Nearest Neighbour Classifier
y_pred, p_res, class_sort, prob_sort = \
                    topk_performance(k_models[0],n_pred,x_train_trans,y_train_trans,x_test_trans,y_test_trans,yencoder)
print('')

##Calculate the best model out of 3 and return results from the best model
ypred_max, p_res_max, class_sort_max, prob_sort_max = \
                    best_topk_model(k_models,n_pred,x_train_trans,y_train_trans,x_test_trans,y_test_trans,yencoder)


df_class_max = pd.DataFrame(data=class_sort_max,columns=['Rank 2', 'Rank 1'])
y_comp = pd.concat([df_class_max,y_test.reset_index(drop=True)],axis=1)
print(y_comp.head(20))


