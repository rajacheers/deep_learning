from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from columnar.models import LinearModel
iris = datasets.load_iris()
X, y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.95, random_state=42)


clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
LinearModel(x_train, x_test, y_train, y_test)