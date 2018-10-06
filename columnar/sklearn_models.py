from data import prepare_data

from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report


x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_data(one_hot=False)

classifiers = [
    GaussianNB(),
    #  RidgeClassifier(tol=1e-2, solver="lsqr"),
    QuadraticDiscriminantAnalysis(),
    LinearDiscriminantAnalysis(),
    DecisionTreeClassifier(max_depth=5),
    KNeighborsClassifier(3, n_jobs=-1),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, n_jobs=-1),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    SVC(kernel="rbf", C=0.025, probability=True),
    MLPClassifier(alpha=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=-1),
]

for clf in classifiers:
    print('_' * 80)
    print(clf.__class__.__name__)
    clf.fit(x_train, y_train)
    print('Train/val/test accuracy: ', clf.score(x_train, y_train), clf.score(x_valid, y_valid), clf.score(x_test, y_test))
    print('Classification report of Test data')
    print(classification_report(y_test, clf.predict(x_test)))
