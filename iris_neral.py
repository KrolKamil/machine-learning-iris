import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

def iris():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pandas.read_csv(url, names=names)

    array = dataset.values
    X = array[:, 0:4]
    Y = array[:, 4]

    validation_size = 0.20
    seed = 7
    # validation = test
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                    random_state=seed)

    # testing:

    results = list()

    # activations = ['identity', 'logistic', 'tanh', 'relu']
    # solvers = ['lbfgs', 'sgd', 'adam']
    # alphas = [0.000001, 0.001, 0.1, 10.0, 1000.0]

    activations = ['identity']
    solvers = ['sgd']
    alphas = [0.001]

    # activations = ['logistic']
    # solvers = ['adam']

    for alpha in alphas:
        for layer in range(1, 6, 1):
            for activation in activations:
                for solver in solvers:
                    mlp = MLPClassifier(hidden_layer_sizes=(layer, layer, layer), activation=activation, solver=solver,
                                        max_iter=800,
                                        verbose=False, tol=1e-4, alpha=alpha)
                    mlp.fit(X_train, Y_train)
                    results.append(
                        (layer, activation, solver, mlp.score(X_train, Y_train), mlp.score(X_validation, Y_validation), alpha))

    results.sort(key=lambda x: (x[4], x[3]), reverse=True)

    return results[:10]


best_methods = list()

for i in range(1, 50, 1):
    for iris_result in iris():
        best_methods.append(iris_result)
    best_methods.sort(key=lambda x: (x[4], x[3]), reverse=True)
    best_methods = best_methods[:10]

for best_method in best_methods:
    print(best_method)

#NOTE LOGISTIC ADAM BETWEEN 25 AND 35 HAS GREATESTS RESULTS
