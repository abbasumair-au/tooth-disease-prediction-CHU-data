import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Import dataset to panda data frame
data1 = pd.read_excel('/home/umair/Downloads/computational_intelligence/data/data1.xlsx')
pd.set_option('display.max_columns', 20)
del data1['Pregnant']
#print(data1)
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

#Voila! Dimensions reduced!
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
#print(type(data1))
#print(data1.loc[0:,'ID':'Sex'])
np_array = data1.to_numpy();
#print(type(np_array))
#print(np_array[:,1:4])
#print(np.shape(np_array))
pca = PCA()
#pca.fit(data1)
X_train, X_test, y_train, y_test = train_test_split(np_array[:,1:14], np_array[:, 15], random_state=0, stratify=np_array[:, 15])

pca = PCA(n_components=13)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca)
print(X_test_pca)
for X, y in zip((X_train_pca, X_test_pca), (y_train, y_test)):
    for i, annot in enumerate(zip(('Gingivitis', 'Periodontitis', 'Healthy'),
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

#next step is clustering.












# #X_train, X_test, y_train, y_test = train_test_split(x_data, y_target, random_state=0, stratify=y_target)
# StandardScaler().fit_transform(np_array)


# # Compute the mean of the data
# mean_vec = np.mean(np_array, axis=0)

# # Compute the covariance matrix
# #cov_mat = (np_array - mean_vec).T.dot((np_array - mean_vec)) / (np_array.shape[0]-1)


# # OR we can do this with one line of numpy:
# cov_mat = np.cov(np_array.T)


# # Compute the eigen values and vectors using numpy
# eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# # Make a list of (eigenvalue, eigenvector) tuples
# eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# # Sort the (eigenvalue, eigenvector) tuples from high to low
# eig_pairs.sort(key=lambda x: x[0], reverse=True)

# # Only keep a certain number of eigen vectors based on 
# # the "explained variance percentage" which tells us how 
# # much information (variance) can be attributed to each 
# # of the principal components

# exp_var_percentage = 0.99 # Threshold of 97% explained variance

# tot = sum(eig_vals)
# var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
# cum_var_exp = np.cumsum(var_exp)

# num_vec_to_keep = 0

# for index, percentage in enumerate(cum_var_exp):
#   if percentage > exp_var_percentage:
#     num_vec_to_keep = index + 1
#     break

# # Compute the projection matrix based on the top eigen vectors
# num_features = np_array.shape[1]
# proj_mat = eig_pairs[0][1].reshape(num_features,1)
# for eig_vec_idx in range(1, num_vec_to_keep):
#   proj_mat = np.hstack((proj_mat, eig_pairs[eig_vec_idx][1].reshape(num_features,1)))

# # Project the data 
# pca_data = np_array.dot(proj_mat)
# print(type(pca_data))
# print(np.shape(pca_data))
# print (pca_data)




# pca.fit(X_train)
# X_train_pca = pca.transform(X_train)
# X_test_pca = pca.transform(X_test)

# for X, y in zip((X_train_pca, X_test_pca), (y_train, y_test)):
#     for i, annot in enumerate(zip(('Gingivitis', 'Periodontitis', 'Healthy'),
#                                   ('blue', 'red', 'green'))):
#         plt.scatter(X[y==i, 0],
#                     X[y==i, 1],
#                     label=annot[0],
#                     c=annot[1])
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')
#     plt.legend(loc='best')
#     plt.tight_layout()
# # plt.show()
                                                    
                                                    
                                                    