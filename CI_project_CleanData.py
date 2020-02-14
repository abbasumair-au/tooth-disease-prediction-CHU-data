# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:07:11 2020

@author: HP
"""
import pandas as pd
data1=pd.read_excel("C:/Users/HP/Desktop/computational intelligence project/data_final2.0.xlsx")
data2 = data1.dropna(axis=0,how='any')
data2.to_excel('C:/Users/HP/Desktop/computational intelligence project/data_final3.0.xlsx')

dataset=pd.read_excel('C:/Users/HP/Desktop/computational intelligence project/data_final3.0.xlsx')
print(dataset.head())
dataset.dtypes
dataset['Sex'] = dataset['Sex'].astype('category')
dataset['Smoking'] = dataset['Smoking'].astype('category')
dataset['Pathologies'] = dataset['Pathologies'].astype('category')
dataset['Food_Sugar'] = dataset['Food_Sugar'].astype('category')
dataset['Fat_Salty'] = dataset['Fat_Salty'].astype('category')
dataset['Soda'] = dataset['Soda'].astype('category')
dataset['Alcohol'] = dataset['Alcohol'].astype('category')
dataset['Frequence_Appoint_Dentist'] = dataset['Frequence_Appoint_Dentist'].astype('category')
dataset['Gingivorrhagia'] = dataset['Gingivorrhagia'].astype('category')
dataset['Diagnosis'] = dataset['Diagnosis'].astype('category')
dataset['Sex'] = dataset['Sex'].cat.codes
dataset['Smoking'] = dataset['Smoking'].cat.codes
dataset['Pathologies'] = dataset['Pathologies'].cat.codes
dataset['Food_Sugar'] = dataset['Food_Sugar'].cat.codes
dataset['Fat_Salty'] = dataset['Fat_Salty'].cat.codes
dataset['Soda'] = dataset['Soda'].cat.codes
dataset['Alcohol'] = dataset['Alcohol'].cat.codes
dataset['Frequence_Appoint_Dentist'] = dataset['Frequence_Appoint_Dentist'].cat.codes
dataset['Gingivorrhagia'] = dataset['Gingivorrhagia'].cat.codes
dataset['Diagnosis'] = dataset['Diagnosis'].cat.codes
dataset.dtypes
dataset=(dataset-dataset.min())/(dataset.max()-dataset.min())
print(dataset.head())
print(dataset.columns)
data_X=dataset.iloc[:,0:15]
print(data_X.columns)
data_Y=dataset.iloc[:,15]
print(data_Y)
#PCA
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
X_train, X_test, y_train, y_test = train_test_split(data_X,data_Y,random_state=0,stratify=data_Y)
print(X_train)
pca=PCA(n_components=2)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
print(X_train_pca)
X_test_pca = pca.transform(X_test)
print(X_test_pca)
for X, y in zip((X_train_pca, X_test_pca), (y_train, y_test)):
    for i,annot in enumerate(zip(('0.0','0.5','1.0'),('blue','red','green'))):
        plt.scatter(X[y==i, 0],
                    X[y==i, 1],
                    label=annot[0],
                    c=annot[1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()







