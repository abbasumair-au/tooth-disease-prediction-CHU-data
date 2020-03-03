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
from sklearn.model_selection import LeaveOneOut
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessClassifier


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

    # bins = [0, 20, 40, 60, 80]
    # labels = [0, 1, 2, 3]
    # data1['Age'] = pd.cut(data1['Age'], bins=bins, labels=labels, right=False)
    # data1['Age'] = data1['Age'].astype(np.int8)
    # print(data1['Age'])

    # bins = [0, 10, 20, 30, 40, 50]
    # labels = [0, 1, 2, 3, 4]
    # data1['BMI'] = pd.cut(data1['BMI'], bins=bins, labels=labels, right=False)
    # data1['BMI'] = data1['BMI'].astype(np.int8)
    # print(data1['BMI'])

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
    data1.iloc[:, 1:15] = (data1.iloc[:, 1:15] - data1.iloc[:, 1:15].min()) / (
            data1.iloc[:, 1:15].max() - data1.iloc[:, 1:15].min())  # data normalization
    # data1=(data1-data1.mean())/data1.std()
    # print(data1.iloc[0:30, :])
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
    # x_pca = data.iloc[:, 1:14]
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
        x_data = data.iloc[:, 1:15]
    if i == 1:
        x_data = data.loc[:, ['Age', 'BMI', 'Pathologies', 'Food_Sugar', 'Hygiene_Dental']]   #use the correlated features
    if i == 2:
        x_data = pca_process()  # using data reduced dimension from PCA
    if i == 3:
        x_data = data.iloc[:, 2]  # use single feature
        x_data = x_data.values.reshape(-1, 1)
    y_data = data.loc[:, ['Diagnosis']]
    return x_data, y_data


def knn_process(i):
    # KNN
    x_data, y_data = data_select(i)
    knn = neighbors.KNeighborsClassifier(n_neighbors=1, weights='uniform')
    #split validation
    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.22, random_state=53)
    knn.fit(X_train, np.ravel(Y_train, order='C'))
    train_score = knn.score(X_train, Y_train)
    test_score = knn.score(X_test, Y_test)
    print('Train Acc: %.3f, Test Acc: %.3f' % (train_score, test_score))
    #K-fold validation
    kfold = model_selection.KFold(n_splits=10)
    results_kfold = model_selection.cross_val_score(knn, x_data, np.ravel(y_data, order='C'), cv=kfold)
    print("Accuracy: %.2f%%" % (results_kfold.mean() * 100.0))
    #leave one out validatoin
    loocv = LeaveOneOut()
    results_loocv = model_selection.cross_val_score(knn, x_data, np.ravel(y_data, order='C'), cv=loocv)
    print("Accuracy: %.2f%%" % (results_loocv.mean() * 100.0))



def decision_tree(i):
    # Decision Tree
    x_data, y_data = data_select(i)
    Dt = DecisionTreeClassifier(max_depth=5)
    #split validation
    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.12, random_state=53)
    Dt.fit(X_train, Y_train)
    train_score = Dt.score(X_train, Y_train)
    test_score = Dt.score(X_test, Y_test)
    print('Train Acc: %.3f, Test Acc: %.3f' % (train_score, test_score))
    #K-fold validation
    kfold = model_selection.KFold(n_splits=10)
    results_kfold = model_selection.cross_val_score(Dt, x_data, np.ravel(y_data, order='C'), cv=kfold)
    print("Accuracy: %.2f%%" % (results_kfold.mean() * 100.0))
    #leave one out validatoin
    loocv = LeaveOneOut()
    results_loocv = model_selection.cross_val_score(Dt, x_data, np.ravel(y_data, order='C'), cv=loocv)
    print("Accuracy: %.2f%%" % (results_loocv.mean() * 100.0))


def support_vector(i):
    x_data, y_data = data_select(i)
    sv = SVC(kernel='linear', gamma='auto')
    #split validation
    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=53)
    sv.fit(X_train, np.ravel(Y_train, order='C'))
    train_score = sv.score(X_train, Y_train)
    test_score = sv.score(X_test, Y_test)
    print('Train Acc: %.3f, Test Acc: %.3f' % (train_score, test_score))
    # K-fold validation
    kfold = model_selection.KFold(n_splits=10)
    results_kfold = model_selection.cross_val_score(sv, x_data, np.ravel(y_data, order='C'), cv=kfold)
    print("Accuracy: %.2f%%" % (results_kfold.mean() * 100.0))
    # leave one out validatoin
    loocv = LeaveOneOut()
    results_loocv = model_selection.cross_val_score(sv, x_data, np.ravel(y_data, order='C'), cv=loocv)
    print("Accuracy: %.2f%%" % (results_loocv.mean() * 100.0))


def neural_network(i):
    x_data, y_data = data_select(i)
    clf = MLPClassifier(hidden_layer_sizes=(4,2), solver='sgd',
                        batch_size=4, learning_rate_init=0.005,
                        max_iter=500, shuffle=True)
    #split validation
    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=53)
    clf.fit(X_train, np.ravel(Y_train, order='C'))
    print("Number of layers: ", clf.n_layers_)
    print("Number of outputs: ", clf.n_outputs_)
    train_score = clf.score(X_train, Y_train)
    test_score = clf.score(X_test, Y_test)
    print('Train Acc: %.3f, Test Acc: %.3f' % (train_score, test_score))
    # K-fold validation
    kfold = model_selection.KFold(n_splits=10)
    results_kfold = model_selection.cross_val_score(clf, x_data, np.ravel(y_data, order='C'), cv=kfold)
    print("Accuracy: %.2f%%" % (results_kfold.mean() * 100.0))
    # leave one out validatoin
    loocv = LeaveOneOut()
    results_loocv = model_selection.cross_val_score(clf, x_data, np.ravel(y_data, order='C'), cv=loocv)
    print("Accuracy: %.2f%%" % (results_loocv.mean() * 100.0))


def GP_Classifier(i):
    x_data, y_data = data_select(i)
    gpc = GaussianProcessClassifier(random_state = 53)
    # split validation
    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=53)
    gpc.fit(X_train, np.ravel(Y_train, order='C'))
    train_score = gpc.score(X_train, Y_train)
    test_score = gpc.score(X_test, Y_test)
    print('Train Acc: %.3f, Test Acc: %.3f' % (train_score, test_score))
    # K-fold validation
    kfold = model_selection.KFold(n_splits=10)
    results_kfold = model_selection.cross_val_score(gpc, x_data, np.ravel(y_data, order='C'), cv=kfold)
    print("Accuracy: %.2f%%" % (results_kfold.mean() * 100.0))
    # leave one out validatoin
    loocv = LeaveOneOut()
    results_loocv = model_selection.cross_val_score(gpc, x_data, np.ravel(y_data, order='C'), cv=loocv)
    print("Accuracy: %.2f%%" % (results_loocv.mean() * 100.0))

if __name__ == '__main__':
    # parameter 0: all the data except id
    # parameter 1: 5 most correlated features
    # parameter 2: data after pca process
    # parameter 3: choose one feature

    # cal_correlation()
    # knn_process(1)
    decision_tree(1)
    # support_vector(1)
    # neural_network(3)
    # GP_Classifier(1)
