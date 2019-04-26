import pandas
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import csv


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv('iris_data.csv', names=names)

array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.20, random_state=7)

mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='logistic', solver='adam', max_iter=1400,
                    verbose=True, tol=1e-4, alpha=1e-06)

mlp.fit(X_train, Y_train)

print(mlp.score(X_validation, Y_validation))

predictions = mlp.predict(X_validation)

print(confusion_matrix(Y_validation, predictions))
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

predicted = mlp.predict(X_validation)
core_list = X_validation.tolist()

for i in range(0, 30, 1):
    core_list[i].append(Y_validation[i])
    core_list[i].append(predicted[i])

core_names = names
core_names[4] = "exact"
core_names.append("predicted")


with open('ssi_result.csv', mode='w') as data:
    data_writer = csv.writer(data)
    data_writer.writerow(core_names)
    for predict in core_list:
        data_writer.writerow(predict)

