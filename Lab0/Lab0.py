
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
#ALGORITHM = "custom_net"
ALGORITHM = "tf_net"


class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)



    # Activation function.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100, minibatches = True, mbs = 100):
        x_val = self.__batchGenerator(xVals, mbs)
        y_val = self.__batchGenerator(yVals, mbs)

        for it in range(epochs):
            x = next(x_val)
            y = next(y_val)
            feed_forward_results = self.__forward(x)

            #layer 2
            layer2error = y - feed_forward_results[1]
            layer2delta = layer2error * self.__sigmoidDerivative(feed_forward_results[1])
            transpose_m = np.transpose(feed_forward_results[0])
            layer2adjustment = np.dot(transpose_m, layer2delta) * self.lr

            #layer1
            layer1error = np.dot(layer2delta, np.transpose(self.W2))
            layer1delta = layer1error * self.__sigmoidDerivative(feed_forward_results[0])
            layer1adjustment = np.dot(np.transpose(x), layer1delta) * self.lr

        self.W1 += layer1adjustment
        self.W2 += layer2adjustment

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2



# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    xTrain = xTrain / 255.0
    yTrain = yTrain / 255.0
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    xTrain = xTrain.reshape([60000, 784])
    xTest = xTest.reshape([10000, 784])
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def tf_model():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()

    xTrain = tf.keras.utils.normalize(xTrain, axis=1)
    xTrain = xTrain.reshape(xTrain.shape[0], -1)

    xTest = tf.keras.utils.normalize(xTest, axis=1)
    xTest = xTest.reshape(xTest.shape[0], -1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    loss = 'sparse_categorical_crossentropy'
    model.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    model.fit(xTrain,yTrain, epochs=40)

    loss, accuracy = model.evaluate(xTest, yTest)
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))


def trainModel(data):
    xTrain, yTrain = data

    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        model = NeuralNetwork_2Layer(inputSize=784,
                                     neuronsPerLayer=256,
                                     outputSize=10)
        model.train(xTrain, yTrain)
        return model
    else:
        raise ValueError("Algorithm not recognized.")

def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        res = model.predict(data)
        return res
    else:
        raise ValueError("Algorithm not recognized.")

def calculate_f1_score(predictions, y_test):
    confusion_m = np.zeros(shape=(10, 10))
    precisions = []
    recalls = []
    for i in range(predictions.shape[0]):
        max_tr = np.argmax(y_test[i])
        max_p = np.argmax(predictions[i])
        confusion_m[max_p, max_tr] += 1

    for i in range(10):
        true_pos = confusion_m[i][i]
        precision = true_pos / (sum(confusion_m[i]))
        precisions.append(precision)

        total = sum([row[i] for row in confusion_m])

        false_neg = total - true_pos
        recall = true_pos / (true_pos + false_neg)
        recalls.append(recall)

    precision_average = sum(precisions) / len(precisions)
    recall_average = sum(recalls) / len(recalls)

    return 2 * (precision_average * recall_average) / (precision_average + recall_average)



def evalResults(data, preds):
    xTest, yTest = data
    acc = 0

    #convert numbers in each array to one hot encoding format related to getting argmax
    for i in range(len(preds)):
        #find max element in array, replace it with 1 and the others as 0
        index = np.argmax(preds[i])
        preds[i] = [0 for x in preds[i]]
        preds[i][index] = 1

    #problem - the max index is always 0 in preds[i], therefore the accuracy is 9,8 all the time, which
    # couldn't understand the reason after training with backpropagation

    for i in range(preds.shape[0]):
        #print(preds[i])
        #print(yTest[i])
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]

    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print("Classifier f1-score: %f%%" % calculate_f1_score(preds,yTest))
    print()



#=========================<Main>================================================

def main():
    if ALGORITHM == 'tf_net':
        return tf_model()

    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
