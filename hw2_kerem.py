import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection as ms
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# homework for week 2


# veri setini kullandık
veri = pd.read_csv("/Users/keremaglik/Documents/Machine Learning Bootcamp/1.Hafta/Week_1 Homework/water_potability.csv")

# eksik veriler dolduruldu
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
veri_filled = veri.iloc[:, :].values
veri_filled_df = pd.DataFrame(data=veri_filled, index=range(3276 ),columns=['pH', 'Hardness', 'Solids', 'Chloramines',
                                                                            'Sulfate', 'Conductivity', 'Organic_carbon',
                                                                            'Trihalomethanes', 'Turbidity', 'Potability'
                                                                            ])

veri_filled_df.to_csv('/Users/keremaglik/Documents/Machine Learning Bootcamp/Week_1 Homework/veriler_filled.csv',
                      index=False)

# x ve y belirlendi
x = veri_filled_df.data[:, :]
y = veri_filled_df.target

# train test split işlemi yapıldı
x_train, x_test, y_train, y_test = ms.train_test_split(x, y, test_size=0.33, random_state=0)

# scale ve fit transform işlemleri yapıldı
sc = StandardScaler()
x_test_ol = sc.fit_transform(x_test)
x_train_ol = sc.fit_transform(x_train)

# SVC sınıflandırması
svc = SVC()
p = [{'C': [1, 2, 3, 4, 5], 'kernel': ['linear']},
     {'C': [1, 2, 3, 4, 5], 'kernel': ['rbf'], 'gamma': [1, 0.5, 0.1, 0.01, 0.001]},
     {'C': [1, 2, 3, 4, 5], 'kernel': ['poly'], 'degree': [1, 2, 3, 4, 5, 6, 7], 'gamma': [1, 0.5, 0.1, 0.01, 0.001]}]

# grid search işlemi yapıldı
gs = ms.GridSearchCV(estimator=svc, param_grid=p, scoring='accuracy', cv=5)
grid_search = gs.fit(x_train_ol, y_train)

eniyiparamat = grid_search.best_params_
eniyiscore = grid_search.best_score_

print("SVC Results \n")
print(eniyiscore)
print(eniyiparamat)

# logistic
logr = LogisticRegression(random_state=0)
logr.fit(x_train_ol, y_train)
y_pred = logr.predict(x_test_ol)

print("Logistic Regression Results \n")
print(y_pred)
print(y_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print("Sınıflandırma raporu \n", metrics.classification_report(y_test, y_pred))
print("doğruluk değeri \n", metrics.accuracy_score(y_test, y_pred))

basari = cross_val_score(estimator=logr, X=x_train_ol, y=y_train, cv=4)

print(basari.mean())
print(basari.std())

# knn K nearest neighbors
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(x_train_ol, y_train)
y_pred_knn = knn.predict(x_test_ol)

print("knn Results \n")
print(y_pred_knn)
print(y_test)

cm_knn = confusion_matrix(y_test, y_pred_knn)
print(cm_knn)
print("Sınıflandırma raporu \n", metrics.classification_report(y_test, y_pred_knn))
print("doğruluk değeri \n", metrics.accuracy_score(y_test, y_pred_knn))

bas_knn = cross_val_score(estimator=knn, X=x_train_ol, y=y_train, cv=4)
print(bas_knn.mean())
print(bas_knn.std())

# Navie Bayes
gnb = GaussianNB()
gnb.fit(x_train_ol, y_train)
y_pred_nb = gnb.predict(x_test_ol)

print("Naive Bayes Results \n")
print(y_test)
print(y_pred_nb)

cm_nbg = confusion_matrix(y_test, y_pred_nb)
print(cm_nbg)

print("Sınıflandırma raporu \n", metrics.classification_report(y_test, y_pred_nb))
print("doğruluk değeri \n", metrics.accuracy_score(y_test, y_pred_nb))

bas_gnb = cross_val_score(estimator=gnb, X=x_train_ol, y=y_train, cv=4)
print(bas_gnb.mean())
print(bas_gnb.std())

# Decision Tree
dtc = DecisionTreeClassifier(criterion='entropy')
dtc2 = DecisionTreeClassifier(criterion='gini')

dtc.fit(x_train_ol, y_train)
y_pred_dtc_en = dtc.predict(x_test_ol)

cm_dtc_en = confusion_matrix(y_test, y_pred_dtc_en)

print("Decision Tree Results \n")
print(y_test)
print(y_pred_dtc_en)
print(cm_dtc_en)

print("Sınıflandırma raporu \n", metrics.classification_report(y_test, y_pred_dtc_en))
print("doğruluk değeri \n", metrics.accuracy_score(y_test, y_pred_dtc_en))

bas_dtc = cross_val_score(estimator=dtc, X=x_train_ol, y=y_train, cv=4)
print(bas_dtc.mean())
print(bas_dtc.std())

dtc2.fit(x_train_ol, y_train)
y_pred_dtc_gini = dtc2.predict(x_test_ol)

cm_dtc_gini = confusion_matrix(y_test, y_pred_dtc_gini)

print(y_test)
print(y_pred_dtc_gini)
print(cm_dtc_gini)

print("Sınıflandırma raporu \n", metrics.classification_report(y_test, y_pred_dtc_gini))
print("doğruluk değeri \n", metrics.accuracy_score(y_test, y_pred_dtc_gini))

bas_dtc2 = cross_val_score(estimator=dtc2, X=x_train_ol, y=y_train, cv=4)
print(bas_dtc2.mean())
print(bas_dtc2.std())

# random forest
for i in range(1, 11):
    rf = RandomForestClassifier(n_estimators=i, criterion='entropy')
    rf.fit(x_train_ol, y_train)
    y_pred_rf = rf.predict(x_test_ol)
    print(y_test)
    print(y_pred_rf)

    print("{} karar ağacı sayısı \n {} Sınıflandırma raporu \n".format(i, metrics.classification_report(y_test,
                                                                                                        y_pred_rf)))
    print("{} karar ağacı sayısı \n {} doğruluk değeri \n".format(i, metrics.accuracy_score(y_test, y_pred_rf)))
    bas_rbf = cross_val_score(estimator=rf, X=x_train_ol, y=y_train, cv=4)
    print(bas_rbf.mean())
    print(bas_rbf.std())

fpr, tpr, thold = metrics.roc_curve(y_test, y_pred_rf, pos_label=0)

print(fpr)
print(tpr)

