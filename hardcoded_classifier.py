# Part 1
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Part 2
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
train_test_split(y, shuffle=False)

# Part 3
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
model = classifier.fit(X_train, y_train)

# Part 4
from sklearn.metrics import accuracy_score
y_predicted = model.predict(X_test)
print(accuracy_score(y_test, y_predicted))

# Part 5
class HardCodedModel:
    def predict(self, X_test):
        if y_test <= 2:
            y_test = 0
            return y_test


class HardCodedClassifier:
    def fit(self, X_train, y_train):
        return HardCodedModel


classifier = HardCodedClassifier()
model = classifier.fit(X_train, y_train)
y_predicted = model.predict(X_test)
print(accuracy_score(y_test, y_predicted))
