import random

class network():
    
    def __init__(self, input_size, hidden_layers, output_size):
        self.in_hidden  = [[random.random() for i in range(input_size)] for j in range(hidden_layers)]
        self.hidden_out = [[random.random() for i in range(hidden_layers)] for j in range(output_size)]
        
        for i in self.in_hidden:
            r = ''
            for j in i:
                r += ('%.2f ' % j)
            print r
        print

        for i in self.hidden_out:
            r = ''
            for j in i:
                r += ('%.2f ' % j)
            print r

def add(x, y):
    """
    Ambiguous addition on integers and numbers
    """
    if isinstance(x, list) and isinstance(y, list):
        return [i + j for i, j in zip(x, y)]
    elif isinstance(y, list):
        return [i + x for i in y]
    elif isinstance(x, list):
        return [i + y for i in x]
    else:
        return x + y
    

def main():
    network(3, 5, 1) 

main()
