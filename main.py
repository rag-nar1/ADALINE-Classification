from Adaline import Adaline
from Image import TrainingData
from GUI import GUI
import numpy as np

def test(adaline, test_image):
    prediction = adaline.predict(test_image)
    if prediction == 1:
        print('Apple')  
    else:
        print('Banana')

def main():
    train = TrainingData() # create the training data object
    apples, bananas = train.main() # load the training data
    X = np.array(apples + bananas) # convert the data to a numpy array
    Y = np.array([1 for _ in range(len(apples))] + [-1 for _ in range(len(bananas))]) # create the labels    
    adaline = Adaline(len(X[0])) # create the adaline model
    adaline.train(X, Y) # train the model

    # create the GUI object
    gui = GUI(adaline)
    # plot the error data
    gui.main(adaline.classification_rate[-1])
    # print the results 

    


if __name__ == '__main__':
    main()