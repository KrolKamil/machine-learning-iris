import pandas
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]

validation_size = 0.20
seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)


mlp = MLPClassifier(hidden_layer_sizes=(5, 5, 5), activation='identity', solver='sgd',
                    max_iter=800,
                    verbose=True, tol=1e-4, alpha=0.001)
mlp.fit(X_validation, Y_validation)

print(mlp.score(X_validation, Y_validation))


