from numpy import transpose, exp, tanh, maximum, square, ones


class UnionGate:
    
    def forward(self, x, w, b):

        self.__x = x
        self.__w = w
        
        return x.dot(w) + b  # Wx + b
    
    def backward(self):

        dx = self.__w
        dw = transpose(self.__x)

        return dx, dw


class Linear:
    
    def forward(self, x):
        
        self.__x = x
        
        return x
    
    def backward(self):
        
        return ones((self.__x.shape[0], self.__x.shape[1]))


class Sigmoid:
          
    def forward(self, x):

        self.__x = x

        return 1/(1 + exp(-x))
    
    def backward(self):

        phi = 1/(1 + exp(-self.__x))  # sigmoid forward

        return (1-phi)*phi
    

class Tanh:
    
    def forward(self, x):

        self.__x = x

        return tanh(x)
    
    def backward(self):

        return 1 - tanh(self.__x)**2


class ReLU:
    
    def forward(self, x):

        self.__x = x

        return maximum(0, x)
    
    def backward(self):

        self.__x[self.__x <= 0] = 0  # 0 if x <= 0
        self.__x[self.__x > 0] = 1  # 1 if x > 0
        
        return self.__x                     


class MeanSquaredError:
    
    def forward(self, x, y):

        self.__x = x

        return (1/x.shape[0]) * square(y-x).sum()  # 1/N * (yd - yp).sum()
    
    def backward(self, y):

        return (2 / self.__x.shape[0]) * (self.__x - y)  # 2/N * (yp - yd) 