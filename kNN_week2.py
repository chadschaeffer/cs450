# load iris data
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

# transform data to scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(X))
print(scaler.mean_)
print(scaler.transform(X))

# create training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2)

# make knn algorithm
def dist(dat1, dat2):
    return (dat1 - dat2)^2

def k(n):
    return (n^(1/2))

def nn(targets, knn):
    return mode(targets)

from statistics import mode
def knnModel(X_train, X_test, k):
    for i in X_test:
        distance = dist(X_train, i)
        knn = k(len(X))
        nn = mode(Y, knn)
    def predict(self, X_test):
        return (Y)

def knnClassifier():
    def fit(X_train, y_train):
        return knnModel()

# make prediction
from sklearn.metrics import accuracy_score
classifier = knnClassifier()
model = classifier.fit(X_train, y_train)
y_predicted = model.predict(X_test)
print(accuracy_score(y_test, y_predicted))

# compare to existing implementation
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)
model = classifier.fit(X_train, y_train)
y_prediction = model.predict(X_test)

# make other prediction
print(accuracy_score(y_test, y_prediction))


