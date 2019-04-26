import pandas
import xlsxwriter
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


class Iris:
    def __init__(self):
        self.x = list()
        self.y = list()
        self.class_names = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']
        self.names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        self.X_train = list()
        self.X_validation = list()
        self.Y_train = list()
        self.Y_validation = list()
        self.mlp = None
        self.predictions = None
        self.dataset = list()

        self.__setUtUpData()

    def __setUtUpData(self):
        self.dataset = pandas.read_csv('iris_data.csv', names=self.names)
        array = self.dataset.values
        self.x = array[:, 0:4]
        self.y = array[:, 4]

        self.X_train, self.X_validation, self.Y_train, self.Y_validation = \
            model_selection.train_test_split(self.x, self.y, test_size=0.20, random_state=7)

    def neuronNetwork(self):
        self.mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='logistic', solver='adam', max_iter=1400,
                                 verbose=True, tol=1e-4, alpha=1e-06)
        self.mlp.fit(self.X_train, self.Y_train)

        self.predictions = self.mlp.predict(self.X_validation)

    def showScores(self):
        score =self.mlp.score(self.X_validation, self.Y_validation)
        print("Percentage accuracy: " + str(round(score * 100, 2)) + "%")

    def excelData(self):
        result_data = self.X_validation.tolist()

        for i in range(0, 30, 1):
            result_data[i].append(self.Y_validation[i])
            result_data[i].append(self.predictions[i])

        core_names = self.names
        core_names[4] = "exact"
        core_names.append("predicted")

        workbook = xlsxwriter.Workbook('final.xlsx')
        worksheet = workbook.add_worksheet("core sheet")

        good = workbook.add_format({'bg_color': '#2be566'})
        bad = workbook.add_format({'bg_color': '#e52b2b'})
        label = workbook.add_format({'bg_color': '#a3a7f7'})

        row = 1
        col = 0

        worksheet.set_row(0, cell_format=label)
        worksheet.write(0, 0, self.names[0])
        worksheet.write(0, 1, self.names[1])
        worksheet.write(0, 2, self.names[2])
        worksheet.write(0, 3, self.names[3])
        worksheet.write(0, 4, "exact")
        worksheet.write(0, 5, "predicted")

        for sl, sw, pl, pw, exact, predicted in result_data:
            if (exact == predicted):
                worksheet.set_row(row, cell_format=good)
            else:
                worksheet.set_row(row, cell_format=bad)
            worksheet.write(row, col, sl)
            worksheet.write(row, col + 1, sw)
            worksheet.write(row, col + 2, pl)
            worksheet.write(row, col + 3, pw)
            worksheet.write(row, col + 4, exact)
            worksheet.write(row, col + 5, predicted)
            row += 1

        workbook.close()

    def confusionMatrix(self):
        title = 'Normalized confusion matrix'
        cm = confusion_matrix(self.Y_validation, self.predictions)
        classes = self.class_names
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

    def sepalPlot(self):
        fig, ax = plt.subplots()
        ax.scatter(self.dataset['sepal-length'], self.dataset['sepal-width'])
        colors = {'Iris-setosa': 'r', 'Iris-versicolor': 'g', 'Iris-virginica': 'b'}
        for i in range(len(self.dataset['sepal-length'])):
            ax.scatter(self.dataset['sepal-length'][i], self.dataset['sepal-width'][i], color=colors[self.dataset['class'][i]])
        ax.set_title('Iris-Sepal Dataset')
        ax.set_xlabel('sepal_length')
        ax.set_ylabel('sepal_width')
        print(colors)

    def petalPlot(self):
        fig, ax = plt.subplots()
        ax.scatter(self.dataset['petal-length'], self.dataset['petal-width'])
        colors = {'Iris-setosa': 'r', 'Iris-versicolor': 'g', 'Iris-virginica': 'b'}
        for i in range(len(self.dataset['petal-length'])):
            ax.scatter(self.dataset['petal-length'][i], self.dataset['petal-width'][i], color=colors[self.dataset['class'][i]])
        ax.set_title('Iris-Petal Dataset')
        ax.set_xlabel('petal_length')
        ax.set_ylabel('petal_width')


myIris = Iris()

myIris.neuronNetwork()
myIris.showScores()
myIris.excelData()
myIris.confusionMatrix()
myIris.sepalPlot()
myIris.petalPlot()

plt.show()
