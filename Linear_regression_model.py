import pandas as pd
import sklearn.model_selection
import numpy as np
import pickle
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

def get_loss(predicted, labels, n):
    loss = 0
    for i in range(len(predicted)):
        loss += ((labels[i] - predicted[i]) ** 2) 
    return loss / n

def compute_gradient(w, x, b, y, n):
    der = 0
    for i in range(len(y)):
        der += (y[i] - np.dot(w, x[i]) - b) * x[i]
    return 2 * der / n

def compute_intercept(w, x, b, y, n):
    der = 0
    for i in range(len(y)):
        der += (y[i] - np.dot(w, x[i]) - b)
    return 2 * der / n

data = pd.read_csv("student-mat.csv", sep = ";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"
x = np.array(data.drop([predict], axis = 1)) #we try to predict G3, so we remove it from the dataset
y = np.array(data[predict]) #we get the labels(actual scores) of G3

#we split the dataset in trainingset and testset(90% training, 10% test)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

try:
    with open("linear_regression_params.pkl", "rb") as f:
        saved_params = pickle.load(f)
        final_slope = saved_params["slope"]
        final_bias = saved_params["bias"]
        print("Loaded saved molde parameters.")
except:
    print("No saved parameters found. Training model...")
    slope = [0] * 5
    bias = 0
    final_slope = slope
    final_bias = bias
    l = 1000000000
    alpha = 0.0001
    for i in range(0, 100000):
        predict = []
        for j in range(len(x_train)):
            predict.append(np.dot(slope, x_train[j]) + bias)
        loss = get_loss(predict, y_train, len(y_train))
        if loss < l:
            l = loss
            final_slope = slope
            final_bias = bias
        gr = compute_gradient(slope, x_train, bias, y_train, len(y_train))
        b = compute_intercept(slope, x_train, bias, y_train, len(y_train))
        slope = slope + alpha * gr
        bias = bias + alpha * b
    print(alpha, loss)
    with open("linear_regression_params.pkl", "wb") as f:
        pickle.dump({"slope": final_slope, "bias": final_bias}, f)
    print("Model parameters saved.")
predict = []
for i in range(len(x_test)):
    predict.append(np.dot(final_slope, x_test[i]) + final_bias)
print(get_loss(predict, y_test, len(y_test)))