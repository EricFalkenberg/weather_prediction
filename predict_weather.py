from random import seed, random
from operator import add, sub, mul, pow, neg
import math

class network():
    
    def __init__(self, f_name, hidden_size):
        seed(10000)
        ## TEST NEURAL NET
        ## XOR
        ## 0 && 0 = 0
        ## 0 && 1 = 1
        ## 1 && 0 = 1
        ## 1 && 1 = 0
        output_size = 1 
        ## Input vector
        self.layer_0 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        ## Output vector
        self.expected_out = [0, 1, 1, 0]
        ## Weights
        self.w0 = [[random() for i in range(hidden_size)] for j in range(len(self.layer_0))]
        self.w_1 = [[random() for i in range(output_size)] for j in range(hidden_size)]
        ## Predictions on test set
        self.predictions = [0 for i in range(len(self.layer_0))]

        ## Learn the model via backpropagation
        for i in range(100000):
            for example in range(len(self.layer_0)):
                ## Get the current examples expected output
                y                  = self.expected_out[example]
                ## Retrieve layers and their sigmoid values
                l0                 = self.layer_0[example]
                l1                 = sigmoid(v_dot(l0, self.w0[example]))
                l2                 = sigmoid(v_dot(l1, self.w_1[example]))
                ## Calculate error and delta in the result layer
                l2_error           = elementwise(sub, ([y for i in range(len(l2))], l2))
                l2_delta           = elementwise(mul, (l2_error, sigmoid(l2, True))) 
                ## Calculate error and delta in the hiddin layer
                l1_error           = v_dot(l2_delta, self.w_1[example])
                l1_delta           = elementwise(mul, (sigmoid(l1, True), l1_error))
                ## Update each weights
                self.w_1[example] = elementwise(add, (self.w_1[example], v_dot(l1, l2_delta)))
                self.w0[example] = elementwise(add, (self.w0[example], v_dot(l0, l1_delta)))
                ## Update the predictions table for the test set
                self.predictions[example] = l2

        ## Print the learned model's predictions for the test set.
        print self.predictions    

def sigmoid(x, deriv=False):
    """
    The sigmoid function our layers use for gradient descent.
    Uses elementwise operations such that it can operate on lists without
    the aid of the numpy library.
    """
    ones = lambda x: [1 for i in range(x)]
    if deriv:
        return elementwise(mul, (x, (elementwise(sub, (ones(len(x)), x)))))
    if not isinstance(x, list):
        arr = [1]
    else:
        arr = ones(len(x))
    return elementwise(pow, (elementwise(add, (arr, elementwise(math.exp, elementwise(neg, x)))), -1))


def elementwise(op, args):
    """
    Take an operation and apply it to the elements in
    the arguments tuple.
    If given lists, apply the operations elementwise.
    """
    if isinstance(args, tuple):
        x, y = args
        if isinstance(y, int) or isinstance(y, float):
            return [op(i, y) for i in x]
        return [op(i, j) for i, j in zip(x, y)]

    return op(args)

def v_dot(x, y):
    """
    Dot product of vector and vector
    """
    return sum([i*j for i, j in zip(x, y)])


def main():
    network('mock', 4) 

main()
