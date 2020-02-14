import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Import dataset to panda data frame
data1 = pd.read_excel("/Users/ethanyang/Desktop/2IS/2IS-M12/Computational_Intelligence/data1.xlsx")
pd.set_option('display.max_columns', 20)
del data1['Pregnant']
data1 = data1.dropna(axis=0, how='any')

#Convert string class to numerical class
data1['Sex'] = data1['Sex'].astype('category')
data1['Sex'] = data1['Sex'].cat.codes
# data1.Sex[data1.Sex == 'Woman'] = 1
# data1.Sex[data1.Sex == 'Man'] = 2

data1['Smoking'] = data1['Smoking'].astype('category')
data1['Smoking'] = data1['Smoking'].cat.codes
# data1.Smoking[data1.Smoking == 'Previous smoker'] = 1
# data1.Smoking[data1.Smoking == 'No smoker'] = 2
# data1.Smoking[data1.Smoking == 'Smoker'] = 3

data1['Pathologies'] = data1['Pathologies'].astype('category')
data1['Pathologies'] = data1['Pathologies'].cat.codes
# data1.Pathologies[data1.Pathologies == 'No'] = 1
# data1.Pathologies[data1.Pathologies == 'Yes'] = 2

# data1.Pregnant.fillna(1, inplace = True)
# list = data1[data1.Sex == 2].index.to_list()
# temp = data1.Pregnant
# print(temp)
# for i in list:
#     temp[i,:].fillna(1, inplace=True)
# print(temp)
# data1.Pregnant = data1[data1.Sex == 2]['Pregnant'].replace(np.nan, 1)
# data1[data1.Sex == 2].fillna({'Pregnant':1},inplace = True)
# fill = pd.Series(1, index = data1[data1.Sex == 2])
# print(fill)
# data1['Pregnant'].fillna(fill, inplace=True)
# print(data1[data1.Sex == 2]['Pregnant'])
# data1['Pregnant'] = data1['Pregnant'].astype('category')
# data1['Pregnant'] = data1['Pregnant'].cat.codes
# data1.Pregnant[data1.Pregnant == 'No'] = 1
# data1.Pregnant[data1.Pregnant == 'Yes'] = 2

data1['Food_Sugar'] = data1['Food_Sugar'].astype('category')
data1['Food_Sugar'] = data1['Food_Sugar'].cat.codes
# data1.Food_Sugar[data1.Food_Sugar == 'Never'] = 1
# data1.Food_Sugar[data1.Food_Sugar == 'Sometimes'] = 2
# data1.Food_Sugar[data1.Food_Sugar == 'Several times a week'] = 3
# data1.Food_Sugar[data1.Food_Sugar == 'Once a day'] = 4
# data1.Food_Sugar[data1.Food_Sugar == 'Several times a day'] = 5

data1['Fat_Salty'] = data1['Fat_Salty'].astype('category')
data1['Fat_Salty'] = data1['Fat_Salty'].cat.codes
# data1.Fat_Salty[data1.Fat_Salty == 'Never'] = 1
# data1.Fat_Salty[data1.Fat_Salty == 'Sometimes'] = 2
# data1.Fat_Salty[data1.Fat_Salty == 'Several times a week'] = 3
# data1.Fat_Salty[data1.Fat_Salty == 'Once a day'] = 4
# data1.Fat_Salty[data1.Fat_Salty == 'Several times a day'] = 5

data1['Soda'] = data1['Soda'].astype('category')
data1['Soda'] = data1['Soda'].cat.codes
# data1.Soda[data1.Soda == 'Never'] = 1
# data1.Soda[data1.Soda == 'Sometimes'] = 2
# data1.Soda[data1.Soda == 'Several times a week'] = 3
# data1.Soda[data1.Soda == 'Once a day'] = 4
# data1.Soda[data1.Soda == 'Several times a day'] = 5

data1['Alcohol'] = data1['Alcohol'].astype('category')
data1['Alcohol'] = data1['Alcohol'].cat.codes
# data1.Alcohol[data1.Alcohol == 'Never'] = 1
# data1.Alcohol[data1.Alcohol == 'Sometimes'] = 2
# data1.Alcohol[data1.Alcohol == 'Several times a week'] = 3
# data1.Alcohol[data1.Alcohol == 'Once a day'] = 4
# data1.Alcohol[data1.Alcohol == 'Several times a day'] = 5

data1['Frequence_Appoint_Dentist'] = data1['Frequence_Appoint_Dentist'].astype('category')
data1['Frequence_Appoint_Dentist'] = data1['Frequence_Appoint_Dentist'].cat.codes
# data1.Frequence_Appoint_Dentist[data1.Frequence_Appoint_Dentist == 'Never'] = 1
# data1.Frequence_Appoint_Dentist[data1.Frequence_Appoint_Dentist == 'Once a year'] = 2
# data1.Frequence_Appoint_Dentist[data1.Frequence_Appoint_Dentist == '2-3 a year'] = 3
# data1.Frequence_Appoint_Dentist[data1.Frequence_Appoint_Dentist == 'Regularly'] = 4

data1['Gingivorrhagia'] = data1['Gingivorrhagia'].astype('category')
data1['Gingivorrhagia'] = data1['Gingivorrhagia'].cat.codes
# data1.Gingivorrhagia[data1.Gingivorrhagia == 'Absent'] = 1
# data1.Gingivorrhagia[data1.Gingivorrhagia == 'Provoked'] = 2
# data1.Gingivorrhagia[data1.Gingivorrhagia == 'Spontanées'] = 3
# data1.Gingivorrhagia[data1.Gingivorrhagia == 'Corrélées (cycle, etc…)'] = 4

data1['Diagnosis'] = data1['Diagnosis'].astype('category')
data1['Diagnosis'] = data1['Diagnosis'].cat.codes
# data1.Diagnosis[data1.Diagnosis == 'Gingivitis'] = 1
# data1.Diagnosis[data1.Diagnosis == 'Periodontitis'] = 2
# data1.Diagnosis[data1.Diagnosis == 'Healthy'] = 3

# print(data1.iloc[0:30,:])
# data1.info()
pca = PCA()
pca = PCA(n_components=2)
x_train = data1.iloc[:,2:4]
y_train = data1.iloc[:,16]
pca.fit(x_train)
# X_train, X_test, y_train, y_test = train_test_split(data1[:,1:15],
#                                                     data1[:, 16],
#                                                     random_state=0,
#                                                     stratify=data1[:, 16])


# X_train_pca = pca.transform(X_train)
# X_test_pca = pca.transform(X_test)
#
for X, y in zip(x_train, y_train):
    for i, annot in enumerate(zip(('Sex', 'Smoking', 'Pathologies'),
                                  ('blue', 'red', 'green'))):
        plt.scatter(X[y==i, 0],
                    X[y==i, 1],
                    label=annot[0],
                    c=annot[1])
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
