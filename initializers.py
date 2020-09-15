from numpy.random import randn
from numpy import sqrt

class Initializers:
    
    __w = []
    __bias = []
        
        
    @classmethod
    def normalized_initialization(cls, initializer, atributes, neurons):
        
        if initializer == 'glorot':
            div = 1
        elif initializer == 'he':
            div = 2
        
        for layer in range(len(neurons)):
            if layer == 0:
                cls.__w.append(randn(atributes, neurons[layer]) / sqrt(atributes / div))  # randn(fan_in, fan_out) / sqrt(fan_in / div)
            else:
                cls.__w.append(randn(neurons[layer-1], neurons[layer]) / sqrt(neurons[layer-1] / div))
            
            cls.__bias.append(randn(1, neurons[layer]) / sqrt(1 / div))
        
        return cls.__w, cls.__bias