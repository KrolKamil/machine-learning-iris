import pandas
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import xlsxwriter


class Iris:
    def __init__(self):
        self.names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        self.X_train = list()
        self.X_validation = list()
        self.Y_train = list()
        self.Y_validation = list()
        self.mlp = None
        self.predictions = None

        self.__setUtUpData()

    def __setUtUpData(self):
        dataset = pandas.read_csv('iris_data.csv', names=self.names)
        array = dataset.values
        x = array[:, 0:4]
        y = array[:, 4]

        self.X_train, self.X_validation, self.Y_train, self.Y_validation = \
            model_selection.train_test_split(x, y, test_size=0.20, random_state=7)

    def neuronNetwork(self):
        self.mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='logistic', solver='adam', max_iter=1400,
                                 verbose=True, tol=1e-4, alpha=1e-06)
        self.mlp.fit(self.X_train, self.Y_train)

        self.predictions = self.mlp.predict(self.X_validation)

    def showScores(self):
        print(self.mlp.score(self.X_validation, self.Y_validation))
        print(confusion_matrix(self.Y_validation, self.predictions))

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

        row = 0
        col = 0

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


myIris = Iris()

myIris.neuronNetwork()
myIris.showScores()
myIris.excelData()

