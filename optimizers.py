from numpy import sqrt, zeros

class Optimizers:
    
    def __init__(self, optmizer, w, n_layers):
        
        self.__optmizer = optmizer
        self.__initialization(w, n_layers)
    
    
    @property
    def optmizer(self):
        return self.__optmizer 
    
    
    def sgd(self, w, grad_w, b, grad_b, lr):

        # x(t+1) = x(t) - lr * dx(t), where x = w or b and dx = gradient
        w -= lr * grad_w  
        b -= lr * grad_b
        
        return w, b
    
    
    def sgd_momentum(self, w, grad_w, b, grad_b, layer, lr, rho):
        
        # velocity(t+1) = momentum * velocity(t) + dx(t), where x = w or b and dx = gradient
        self.__velocity_w[layer] = rho * self.__velocity_w[layer] + grad_w
        self.__velocity_b[layer] = rho * self.__velocity_b[layer] + grad_b
        
        # x(t+1) = x(t) - lr * velocity(t+1), where x = w or b and dx = gradient
        w -= lr * self.__velocity_w[layer]
        b -= lr * self.__velocity_b[layer]
        
        return w, b
    
    
    def ada_grad(self, w, grad_w, b, grad_b, layer, lr):
        
        # dx_squared(t+1) = dx_squared(t) + dx(t) ** 2, where x = w or b and dx = gradient
        self.__grad_squared_w[layer] += grad_w * grad_w
        self.__grad_squared_b[layer] += grad_b * grad_b
        
        # x(t+1) = x(t) - lr * dx(t) / sqrt(dx_squared(t+1) + epsilon), where x = w or b and dx = gradient
        w -= lr * grad_w / (sqrt(self.__grad_squared_w[layer]) + 1e-7)
        b -= lr * grad_b / (sqrt(self.__grad_squared_b[layer]) + 1e-7)
        
        return w, b
    
    
    def rmsprop(self, w, grad_w, b, grad_b, layer, lr, rho):
        
        # dx_squared(t+1) = decay_rate * dx_squared(t) + (1 - decay_rate) * dx(t) ** 2, where x = w or b and dx = gradient
        self.__grad_squared_w[layer] = rho * self.__grad_squared_w[layer] + (1 - rho) * grad_w * grad_w
        self.__grad_squared_b[layer] = rho * self.__grad_squared_b[layer] + (1 - rho) * grad_b * grad_b
        
        w -= lr * grad_w / (sqrt(self.__grad_squared_w[layer]) + 1e-7)
        b -= lr * grad_b / (sqrt(self.__grad_squared_b[layer]) + 1e-7)
        
        return w, b
    
    
    def adam(self, w, grad_w, b, grad_b, layer, lr, rho1, rho2, t):
        
        # first and second moment
        self.__first_moment_w[layer] = rho1 * self.__first_moment_w[layer] + (1 - rho1) * grad_w
        self.__second_moment_w[layer] = rho2 * self.__second_moment_w[layer] + (1 - rho2) * grad_w * grad_w
        
        self.__first_moment_b[layer] = rho1 * self.__first_moment_b[layer] + (1 - rho1) * grad_b
        self.__second_moment_b[layer] = rho2 * self.__second_moment_b[layer] + (1 - rho2) * grad_b * grad_b
    
        # Bias correction
        first_unbias_w = self.__first_moment_w[layer] / (1 - rho1 ** t)
        second_unbias_w = self.__second_moment_w[layer] / (1 - rho2 ** t)
        
        first_unbias_b = self.__first_moment_b[layer] / (1 - rho1 ** t)
        second_unbias_b = self.__second_moment_b[layer] / (1 - rho2 ** t)
        
        # Updating
        w -= lr * first_unbias_w / (sqrt(second_unbias_w) + 1e-7)
        b -= lr * first_unbias_b / (sqrt(second_unbias_b) + 1e-7)
        
        return w, b
    
    
    def __initialization(self, w, n_layers):
        
        # Initializing some parameters of the optimizers
        if self.__optmizer == 'sgd_momentum':
            self.__velocity_w = []
            self.__velocity_b = []
            
            for layer in range(n_layers):
                self.__velocity_w.append(zeros((len(w[layer]), len(w[layer][0]))))
                self.__velocity_b.append(zeros((1, len(w[layer][0]))))
                
        elif self.__optmizer == 'adagrad' or self.__optmizer == 'rmsprop':
            self.__grad_squared_w = []
            self.__grad_squared_b = []
            
            for layer in range(n_layers):
                self.__grad_squared_w.append(zeros((len(w[layer]), len(w[layer][0]))))
                self.__grad_squared_b.append(zeros((1, len(w[layer][0]))))
        
        elif self.__optmizer == 'adam':
            self.__first_moment_w = []
            self.__second_moment_w = []
            self.__first_moment_b = []
            self.__second_moment_b = []
            
            for layer in range(n_layers):
                self.__first_moment_w.append(zeros((len(w[layer]), len(w[layer][0]))))
                self.__second_moment_w.append(zeros((len(w[layer]), len(w[layer][0]))))
                
                self.__first_moment_b.append(zeros((1, len(w[layer][0]))))
                self.__second_moment_b.append(zeros((1, len(w[layer][0]))))