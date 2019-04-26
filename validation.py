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
import pandas
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]

activations = ['identity', 'logistic', 'tanh', 'relu']
solvers = ['lbfgs', 'sgd', 'adam']
alphas = [0.000001, 0.001, 0.1, 10.0, 1000.0]

results = list()

def iris_test():
    for activation in activations:
        for solver in solvers:
            for alpha in alphas:
                mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation=activation, solver=solver, max_iter=1200,
                                    verbose=False, tol=1e-4, alpha=alpha)
                kfold = model_selection.KFold(n_splits=10, random_state=7)
                cv_results = model_selection.cross_val_score(mlp, X, Y, cv=kfold, scoring='accuracy')

                results.append((activation, solver, alpha, cv_results.mean(), cv_results.std()))

    results.sort(key=lambda x: x[3], reverse=True)
    return results[:10]


core_results = list()

for i in range(0, 11, 1):
    core_results = core_results + iris_test()

core_results.sort(key=lambda x: x[3], reverse=True)

for i in range(0, 11, 1):
    print(core_results)


# mlp.fit(X, Y)
# print(mlp.n_layers_ )

# validation_size = 0.20
# seed = 7
#
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
#                                                                                 random_state=seed)
# mlp.fit(X_train, Y_train)
# print(mlp.score(X_validation, Y_validation))