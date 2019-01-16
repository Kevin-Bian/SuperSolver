from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib




data =  pd.read_csv("/Users/KevinBian/projects/opencv-sudoku/mnist_train.csv").as_matrix()
training_data = data[0:40000,1:]
training_label = data[0:40000,0]

test_data =  pd.read_csv("/Users/KevinBian/projects/opencv-sudoku/mnist_test.csv").as_matrix()
testing_data = test_data[0:40000,1:]
testing_label = test_data[0:40000,0]



knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(training_data, training_label)
score = knn.score(testing_data, testing_label)

print("The accuracy is: " + str(score))

joblib.dump(knn, "classifier.pkl", compress=3)