import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def data_preprocess():
    # Import dataset to panda data frame
    data1 = pd.read_excel("/Users/ethanyang/Desktop/2IS/2IS-M12/Computational_Intelligence/data1.xlsx")
    pd.set_option('display.max_columns', 20)
    del data1['Pregnant']
    data1 = data1.dropna(axis=0, how='any')

    # Convert string class to numerical class
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
    data1.iloc[:, 0:14] = (data1.iloc[:, 0:14] - data1.iloc[:, 0:14].min()) / (
            data1.iloc[:, 0:14].max() - data1.iloc[:, 0:14].min())  # data normalization
    # data1=(data1-data1.mean())/data1.std()

    return data1


def cal_correlation():
    # Calculate the correlation
    data = data_preprocess()
    corr = data.corr()
    # print(corr)
    # sns.heatmap(corr, annot=True, fmt=".1f")
    # plt.show()


def pca_process():
    # PCA
    data = data_preprocess()
    pca = PCA()
    pca = PCA(n_components=1)
    x_pca = data.loc[:, ['Age', 'BMI', 'Pathologies', 'Food_Sugar', 'Hygiene_Dental']]
    y_pca = data.loc[:, ['Diagnosis']]
    # X_train, X_test, y_train, y_test = train_test_split(x_data, y_target, random_state=0, stratify=y_target)
    pca.fit(x_pca)
    X_train_pca = pca.transform(x_pca)
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
    #     plt.show()
    return X_train_pca


def data_select(i):
    data = data_preprocess()
    if i == 0:
        x_data = data.iloc[:, 1:14]
    if i == 1:
        x_data = data.loc[:, ['Age', 'BMI', 'Pathologies', 'Food_Sugar', 'Hygiene_Dental']]
    if i == 2:
        x_data = pca_process()  # using data reduced dimension from PCA
    y_data = data.loc[:, ['Diagnosis']]
    return x_data, y_data


def knn_process(i):
    # KNN
    x_data, y_data = data_select(i)
    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=53)
    knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train, np.ravel(Y_train, order='C'))
    train_score = knn.score(X_train, Y_train)
    test_score = knn.score(X_test, Y_test)
    print('Train Acc: %.3f, Test Acc: %.3f' % (train_score, test_score))


def decision_tree(i):
    # Decision Tree
    x_data, y_data = data_select(i)
    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=53)
    Dt = DecisionTreeClassifier(max_depth=4).fit(X_train, Y_train)
    train_score = Dt.score(X_train, Y_train)
    test_score = Dt.score(X_test, Y_test)
    print('Train Acc: %.3f, Test Acc: %.3f' % (train_score, test_score))


def support_vector(i):
    x_data, y_data = data_select(i)
    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=53)
    sv = SVC(kernel='rbf', gamma='auto').fit(X_train, np.ravel(Y_train, order='C'))
    train_score = sv.score(X_train, Y_train)
    test_score = sv.score(X_test, Y_test)
    print('Train Acc: %.3f, Test Acc: %.3f' % (train_score, test_score))


def neural_network(i):
    x_data, y_data = data_select(i)
    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=53)
    clf = MLPClassifier(hidden_layer_sizes=(4,2), solver='sgd',
                        batch_size=4, learning_rate_init=0.005,
                        max_iter=500, shuffle=True)
    clf.fit(X_train, np.ravel(Y_train, order='C'))
    print("Number of layers: ", clf.n_layers_)
    print("Number of outputs: ", clf.n_outputs_)
    train_score = clf.score(X_train, Y_train)
    test_score = clf.score(X_test, Y_test)
    print('Train Acc: %.3f, Test Acc: %.3f' % (train_score, test_score))


if __name__ == '__main__':
    # parameter 0: all the data except id
    # parameter 1: 5 most correlated features
    # parameter 2: data after pca process

    # knn_process(1)
    # decision_tree(1)
    # support_vector(1)
    neural_network(1)
