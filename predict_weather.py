import sys
import pickle
from random import seed, random
from operator import add, sub, mul, pow, neg
import math
import csv
import time

class network():
    
    def get_data(self, f_name):
        """
        Read a given dataset from the file name specified.
        """
        ret = []
        skip = True
        with open(f_name) as csvfile:
            f_read = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in f_read:
                if skip:
                    skip = False
                    continue
                ret.append([float(i) for i in row])
        ret.pop(0)
        return ret

    def __init__(self, test_in, test_out, hidden_size):
        """
        Initialize the neural network.
        """
        ## Initialize constants and random seed.
        seed(10000)
        output_size = 1
        learning_rate = 0.2
        error_threshold = 0.11
        IN   = 0
        HIDE = 1
        OUT  = 2       
        err = [1000 for i in range(len(training_set))]
        ## Input vector
        data = self.get_data(test_in) 
        training_set = data[:2*len(data)//3]
        test_set     = data[2*len(data)//3:]
        ## Output vector
        data = self.get_data(test_out)
        training_out = data[:2*len(data)//3]
        test_out     = data[2*len(data)//3:]
        ## Weights
        w_0 = [[random() for i in range(hidden_size)] for j in range(len(training_set[0]))]
        w_1 = [[random() for i in range(output_size)] for j in range(hidden_size)]

        ## Learn the model via backpropagation
        print "Training network..."
        while sum(err)/len(err) > error_threshold:
            err = []
            ## For every example in the training set
            for example in range(len(training_set)):
                ## Example input
                x = training_set[example]
                ## Example output
                y = training_out[example]
                ## Layer inputs
                in_arr   = [[], [], []]
                ## Layer actuators
                a_arr    = [[], [], []]
                ## Layer deltas
                delt_arr = [[], [], []] 

                ## Propagate the inputs forward
                for i in training_set[example]:
                    a_arr[IN].append(i)
                    in_arr[IN].append(i)
                
                for j in range(hidden_size):
                    in_arr[HIDE].append(sum([w_0[i][j]*a_arr[IN][i] for i in range(len(w_0))]))
                    a_arr[HIDE].append(sigmoid(in_arr[HIDE][j]))
                
                for j in range(output_size):
                    in_arr[OUT].append(sum([w_1[i][j]*a_arr[HIDE][i] for i in range(len(w_1))]))
                    a_arr[OUT].append(sigmoid(in_arr[OUT][j]))

                ## Back propagation of deltas        
                for j in range(output_size):
                    delt_arr[OUT].append(sigmoid(in_arr[OUT][j], deriv=True) * (y[j] - a_arr[OUT][j]))

                ## Record the example error
                err.append( abs( y[0] - a_arr[OUT][0] ) )

                for i in range(hidden_size):
                    delt_arr[HIDE].append(sigmoid(in_arr[HIDE][i], deriv=True)*(sum([w_1[i][j]*delt_arr[OUT][j] for j in range(len(delt_arr[OUT]))])))
                
                for i in range(len(x)):
                    delt_arr[IN].append(sigmoid(in_arr[IN][i], deriv=True)*(sum([w_0[i][j]*delt_arr[HIDE][j] for j in range(len(delt_arr[HIDE]))]))) 

                ## Update weights
                for i in range(len(w_0)):
                    for j in range(len(w_0[i])):
                        w_0[i][j] = w_0[i][j] + learning_rate * a_arr[IN][i] * delt_arr[HIDE][j]
                for i in range(len(w_1)):
                    for j in range(len(w_1[i])):
                        w_1[i][j] = w_1[i][j] + learning_rate * a_arr[HIDE][i] * delt_arr[OUT][j]

                ## Save values
                self.w_0 = w_0
                self.w_1 = w_1
                self.hidden_size = hidden_size
                self.output_size = output_size
                self.training_set = training_set
                self.test_set = test_set
                self.training_out = training_out
                self.test_out = test_out

    
    def predict(self, x):
        ## Using data about a certain day in vector x, predict the weather for
        ## the day after x
        IN   = 0
        HIDE = 1
        OUT  = 2       
        in_arr   = [[], [], []]
        a_arr    = [[], [], []]
        delt_arr = [[], [], []] 

        for i in x: 
            a_arr[IN].append(i)
            in_arr[IN].append(i)
        
        for j in range(self.hidden_size):
            in_arr[HIDE].append(sum([self.w_0[i][j]*a_arr[IN][i] for i in range(len(self.w_0))]))
            a_arr[HIDE].append(sigmoid(in_arr[HIDE][j]))
        for j in range(self.output_size):
            in_arr[OUT].append(sum([self.w_1[i][j]*a_arr[HIDE][i] for i in range(len(self.w_1))]))
            a_arr[OUT].append(sigmoid(in_arr[OUT][j]))

        classification = -1            
        dist = 999
        for i in [0.0, 0.333, 0.666, 1.0]:
            if abs(i - a_arr[OUT][0]) < dist:
                dist = a_arr[OUT][0]
                classification = i
        if classification == 0.0:
            print "It will neither rain nor snow"
        if classification == 0.333:
            print "It will rain"
        if classification == 0.666:
            print "It will snow"
        if classification == 1.0:
            print "It will both rain and snow"

    def testtraining(self):
        ## Test the neural network against the training set
        IN   = 0
        HIDE = 1
        OUT  = 2       
        correct = 0
        incorrect = 0
        for example in range(len(self.training_set)):
            x = self.training_set[example]
            y = self.training_out[example]

            in_arr   = [[], [], []]
            a_arr    = [[], [], []]
            delt_arr = [[], [], []] 

            for i in x: 
                a_arr[IN].append(i)
                in_arr[IN].append(i)
            
            for j in range(self.hidden_size):
                in_arr[HIDE].append(sum([self.w_0[i][j]*a_arr[IN][i] for i in range(len(self.w_0))]))
                a_arr[HIDE].append(sigmoid(in_arr[HIDE][j]))
            for j in range(self.output_size):
                in_arr[OUT].append(sum([self.w_1[i][j]*a_arr[HIDE][i] for i in range(len(self.w_1))]))
                a_arr[OUT].append(sigmoid(in_arr[OUT][j]))

            classification = -1            
            dist = 999
            for i in [0.0, 0.333, 0.666, 1.0]:
                if abs(i - a_arr[OUT][0]) < dist:
                    dist = a_arr[OUT][0]
                    classification = i
            if classification == y[0]:
                correct += 1
            else:
                incorrect += 1
        print "%.2f percent of training tests are passing." % (100.0*float(correct)/float(correct+incorrect))

    def testset(self):
        ## Test the neural network against the test set
        IN   = 0
        HIDE = 1
        OUT  = 2       
        correct = 0
        incorrect = 0
        for example in range(len(self.test_set)):
            x = self.test_set[example]
            y = self.test_out[example]

            in_arr   = [[], [], []]
            a_arr    = [[], [], []]
            delt_arr = [[], [], []] 

            for i in x: 
                a_arr[IN].append(i)
                in_arr[IN].append(i)
            
            for j in range(self.hidden_size):
                in_arr[HIDE].append(sum([self.w_0[i][j]*a_arr[IN][i] for i in range(len(self.w_0))]))
                a_arr[HIDE].append(sigmoid(in_arr[HIDE][j]))
            for j in range(self.output_size):
                in_arr[OUT].append(sum([self.w_1[i][j]*a_arr[HIDE][i] for i in range(len(self.w_1))]))
                a_arr[OUT].append(sigmoid(in_arr[OUT][j]))

            classification = -1            
            dist = 999
            for i in [0.0, 0.333, 0.666, 1.0]:
                if abs(i - a_arr[OUT][0]) < dist:
                    dist = a_arr[OUT][0]
                    classification = i
            if classification == y[0]:
                correct += 1
            else:
                incorrect += 1
        print "%.2f percent of test set tests are passing." % (100.0*float(correct)/float(correct+incorrect))


def sigmoid(x, deriv=False):
    """
    The sigmoid function our layers use for gradient descent.
    Uses elementwise operations such that it can operate on lists without
    the aid of the numpy library.
    """
    if deriv:
        return math.e**x/(((math.e**x) + 1)**2)
    else:
        return 1/(1+math.e**(-1*x))


def main():
    args = sys.argv
    if len(args) != 2:
        print "usage: python predict_weather.py [train] [testtraining] [predict]"
        return    
    if args[1] == 'train':
        nn = network('test_in.csv', 'test_out.csv', 6) 
        print "Network has been trained: "
        nn.testtraining()
        nn.testset()
    else:
        with open('learned_model', 'rb') as f:
            nn = pickle.load(f)
    if args[1] == 'testtraining':
        nn.testtraining()
    if args[1] == "predict":
        print "Please input the following information. Make sure to standardize inputs before attempting prediction"
        prcp = float(raw_input("PRCP: "))
        snwd = float(raw_input("SNWD: "))
        snow = float(raw_input("SNOW: ")) 
        tmax = float(raw_input("TMAX: "))
        tmin = float(raw_input("TMIN: "))
        awnd = float(raw_input("AWND: "))
        wdf2 = float(raw_input("WDF2: "))
        wdf5 = float(raw_input("WDF5: "))
        wsf2 = float(raw_input("WSF2: "))
        wsf5 = float(raw_input("WSF5: "))
        nn.predict([prcp, snwd, snow, tmax, tmin, awnd, wdf2, wdf5, wsf2, wsf5])
    
main()
