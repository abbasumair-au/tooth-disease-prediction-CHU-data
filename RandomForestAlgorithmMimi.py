import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Import dataset to panda data frame
dataset = pd.read_excel("data1.xlsx")
pd.set_option('display.max_columns', 20)
dataset = dataset.dropna(axis=0, how='any')



# Convert string class to numerical class
dataset['Sex'] = dataset['Sex'].astype('category')
dataset['Sex'] = dataset['Sex'].cat.codes

dataset['Smoking'] = dataset['Smoking'].astype('category')
dataset['Smoking'] = dataset['Smoking'].cat.codes
    
dataset['Pathologies'] = dataset['Pathologies'].astype('category')
dataset['Pathologies'] = dataset['Pathologies'].cat.codes

dataset['Pregnant'] = dataset['Pregnant'].astype('category')
dataset['Pregnant'] = dataset['Pregnant'].cat.codes

dataset['Food_Sugar'] = dataset['Food_Sugar'].astype('category')
dataset['Food_Sugar'] = dataset['Food_Sugar'].cat.codes

dataset['Fat_Salty'] = dataset['Fat_Salty'].astype('category')
dataset['Fat_Salty'] = dataset['Fat_Salty'].cat.codes

dataset['Soda'] = dataset['Soda'].astype('category')
dataset['Soda'] = dataset['Soda'].cat.codes

dataset['Alcohol'] = dataset['Alcohol'].astype('category')
dataset['Alcohol'] = dataset['Alcohol'].cat.codes
    
dataset['Frequence_Appoint_Dentist'] = dataset['Frequence_Appoint_Dentist'].astype('category')
dataset['Frequence_Appoint_Dentist'] = dataset['Frequence_Appoint_Dentist'].cat.codes

dataset['Gingivorrhagia'] = dataset['Gingivorrhagia'].astype('category')
dataset['Gingivorrhagia'] = dataset['Gingivorrhagia'].cat.codes

dataset['Diagnosis'] = dataset['Diagnosis'].astype('category')
dataset['Diagnosis'] = dataset['Diagnosis'].cat.codes

#print(dataset.head())

#Prepare data for Training, I don't want to evaluate ID so I start in 1, not in 0.

X = dataset.iloc[:, 1:16].values
y = dataset.iloc[:, 16].values

#divide data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=0)


# Feature Scaling (Scale our data, not compulsory)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Training to solve the regression

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0) #n_estimators defines the number of trees in the random
# forest and random state Controls both the randomness of the bootstrapping of the samples used when building trees
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


#Training using Random Forest Classification that is what we wanted.

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

train_pred = regressor.predict(X_train)

#Evaluating - TEST
print(regressor.score(X_test, y_test))

print("Accuracy: %.2f%%" % (r2_score(y_train, train_pred) * 100))

