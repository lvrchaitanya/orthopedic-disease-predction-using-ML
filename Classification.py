import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, metrics, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# DATESET1
ti1 = pd.read_csv("dataset_1.csv")
p1 = preprocessing.LabelEncoder()
y1 = p1.fit_transform(list(ti1["class"]))
X1 = ti1[["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle", "sacral_slope", "pelvic_radius", "degree"]]
Y1 = list(y1)
x1_train, x1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size=0.2)
arr1 = ["normal", "abnormal"]

# DATASET2
ti2 = pd.read_csv("dataset_2.csv")
p2 = preprocessing.LabelEncoder()
y2 = p2.fit_transform(list(ti2["class"]))
X2 = ti2[["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle", "sacral_slope", "pelvic_radius", "degree"]]
Y2 = list(y2)
x2_train, x2_test, y2_train, y2_test = train_test_split(X2, Y2, test_size=0.3)
arr2 = ["Hernia", "Spondylolisthesis", "normal"]

# HeatMap
datavis= ti1
datavis['class']=y1
# plt.figure(figsize=(15, 8))
# sns.heatmap(datavis.corr(), annot=True)
# plt.savefig("Heat_map")
# plt.show()

# #BoxPlot
# ax = sns.boxplot(data=datavis)
# plt.show()


# model for detecting Abnormality
def model1(model):
    model.fit(x1_train, y1_train)
    predicted = model.predict(x1_test)

    for i in range(len(x1_test)):
        if predicted[i] == 1:
            model2(x1_test.values[i])

    acc = metrics.accuracy_score(y1_test, predicted)
    return str(acc)


# model for predecting specific disease
def model2(x_test):
    x_test = x_test.reshape((6, 1))
    model = svm.SVC(kernel="poly", C=3)
    # model = RandomForestClassifier(n_estimators=50)
    # model = GradientBoostingClassifier(n_estimators=200, random_state=1, learning_rate=0.1)
    model.fit(x2_train, y2_train)
    pre = model.predict(x2_test)
    acc2 = metrics.accuracy_score(y2_test, pre)
    predicted = model.predict(x_test.T)
    print("pridcted :", arr2[predicted[0]], " with Accuracy " + str(acc2))


# to print the Scores of model
def accScores():

    model = svm.SVC(kernel="poly", C=3)
    # model = GradientBoostingClassifier(n_estimators=200, random_state=1, learning_rate=0.1)
    model.fit(x1_train, y1_train)
    pre = model.predict(x1_test)
    acc = metrics.accuracy_score(y1_test, pre)
    cm = confusion_matrix(y1_test, pre)
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[1][1]
    specificity = tn / (tn + fp)
    sensitivity = tp / (tn + fp)
    precision = tp / (tp + fp)
    recall = tp / (fn + tp)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("\nconfusion matrix \n" + str(cm))
    print("specificity " + str(specificity))
    print("sensitivit "+str(sensitivity))
    print("recall "+str(recall))
    print("precision "+str(precision))
    print("f1_score "+str(f1_score))



# print("SVM=" + model1(svm.SVC(kernel="poly", C=2)))
# print("LogisticRegression=" + model1(LogisticRegression()))
print("KNeighborsClassifier=" + model1(KNeighborsClassifier(n_neighbors=5)))
#
# GBmodel = GradientBoostingClassifier(n_estimators=200, random_state=1 , learning_rate=0.1)
# print("RandomForestClassifie=" + model1(RandomForestClassifier(n_estimators=30)))
# print("LogisticRegression=" + model1(GBmodel))
accScores()
