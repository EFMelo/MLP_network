from numpy import transpose, zeros
from numpy.random import shuffle
import functions
from initializers import Initializers
from optimizers import Optimizers


class MLP:

    """
    Description
    -----------
    MLP Configuration.

    Parameters
    ----------
    input_shape : tuple or list
                 Must have two elements: number of samples and number of attributes.
    neurons : tuple or list
             The elements must contain the number of neurons in each layer.
    activations : tuple or list
                 The elements must contain the activation functions in each layer.
    initializer : str, optional
                 Weights and bias initialization.
                 The 'glorot' or 'he' arguments can be entered.
    """

    
    def __init__(self, input_shape, layers, activations, initializer='glorot'):
        
        self.__n_images = input_shape[0]  # number of images
        self.__n_atributes = input_shape[1]  # number of atributes
        self.__n_layers = len(layers)  # number of layers
        self.__funtions_init(activations)  # activation funtions on each layern
        self.__loss = functions.MeanSquaredError()  # MSE Loss
        self.__w, self.__b = Initializers.normalized_initialization(initializer, self.__n_atributes, layers)  # Iniciating weights and bias
        
        
    def run(self, x, y, epochs, batch_size=0, lr=0.001, optimizer='adam', rho1=0.9, rho2=0.999, shuffle=True):

        """
        Description
        -----------
        MLP Training.

        Parameters
        ----------
        x : ndarray
        y : ndarray
        epochs : int
        batch_size : int
        optmizer : str, optional
                  Arguments - 'sgd', 'sgd_momentum', 'adagrad', 'rmsprop' or 'adam'.
        lr : float.
        rho1 : float, optional
              first moment
        rho2 : float, optional
              second moment
        shuffle : bool, optional
        
        Returns
        -------
        history : list
                 Loss funcion values throughout the training.
        """
        

        # if the batch_size was not informed
        if batch_size == 0:
            batch_size = self.__n_images

        # initializing the optimizer  
        opt = Optimizers(optimizer, self.__w, self.__n_layers)
        

        history = []
        # Running the epochs
        for i in range(1, epochs+1):
            total_loss = 0
            
            # shuffling the data
            if shuffle:
                x, y = self.__shuffling_data(x, y)

            # indexes for get samples of set of training images
            ini_batch = 0
            end_batch = batch_size
            
            # interval of batches until reach n_images
            while ini_batch < self.__n_images:
                
                # Forward pass
                s = self.__forward_pass(x[ini_batch:end_batch])
                
                # Calculating the Train loss                
                total_loss += self.__loss.forward(s, y[ini_batch:end_batch])
                
                # Backward pass
                self.__backward_pass(self.__loss, y[ini_batch:end_batch], opt, lr, rho1, rho2, i)

                # updating the indexes of batch
                ini_batch += batch_size
                end_batch += batch_size
                
                if end_batch > self.__n_images:
                    end_batch = self.__n_images

            # loss history     
            print(f'Train_Loss [{i}]: {round(total_loss/self.__n_images, 6)}')
            history.append(total_loss/self.__n_images)

        return history
    
    
    def predict(self, x):
        return self.__forward_pass(x)
    
    
    def __forward_pass(self, x_batch):
        
        s = self.__unions_gate[0].forward(x_batch, self.__w[0], self.__b[0])  # score = W*x + b
        s = self.__activations[0].forward(s)  # score = f(s)
             
        # Loop in layers beginning in the layer 2
        for layer in range(1, self.__n_layers):
            s = self.__unions_gate[layer].forward(s, self.__w[layer], self.__b[layer])  # score = W*score + b
            s = self.__activations[layer].forward(s)  # score = f(s)
                    
        return s
    
    
    def __backward_pass(self, loss, y_batch, opt, lr, rho1, rho2, epoch):
        
        grad_coming = self.__activations[self.__n_layers-1].backward() * self.__loss.backward(y_batch)
                
        for layer in range(self.__n_layers-1, -1, -1):

            grad_s, grad_w = self.__unions_gate[layer].backward()  # local gradients
                    
            # gradients of s and w
            grad_s = transpose(grad_s.dot(transpose(grad_coming)))
            grad_w = grad_w.dot(grad_coming)
            grad_b = grad_coming.sum(axis=0)
                    
            # updating w, bias
            if opt.optmizer == 'sgd':
                self.__w[layer], self.__b[layer] = opt.sgd(self.__w[layer], grad_w, self.__b[layer], grad_b, lr)
            elif opt.optmizer == 'sgd_momentum':
                self.__w[layer], self.__b[layer] = opt.sgd_momentum(self.__w[layer], grad_w, self.__b[layer], grad_b, layer, lr, rho1)
            elif opt.optmizer == 'adagrad':
                self.__w[layer], self.__b[layer] = opt.ada_grad(self.__w[layer], grad_w, self.__b[layer], grad_b, layer, lr)
            elif opt.optmizer == 'rmsprop':
                self.__w[layer], self.__b[layer] = opt.rmsprop(self.__w[layer], grad_w, self.__b[layer], grad_b, layer, lr, rho1)
            elif opt.optmizer == 'adam':
                self.__w[layer], self.__b[layer] = opt.adam(self.__w[layer], grad_w, self.__b[layer], grad_b, layer, lr, rho1, rho2, epoch)

            # updating gradient coming
            if layer > 0:
                grad_coming = self.__activations[layer-1].backward() * grad_s
            
    
    def __funtions_init(self, activations):

        self.__activations = []
        self.__unions_gate = []
        
        # Initializing the activation functions
        for i in range(len(activations)):
            self.__unions_gate.append(functions.UnionGate())
            
            if activations[i] == 'sigmoid':
                self.__activations.append(functions.Sigmoid())
            elif activations[i] == 'tanh':
                self.__activations.append(functions.Tanh())
            elif activations[i] == 'relu':
                self.__activations.append(functions.ReLU())

    
    def __shuffling_data(self, x, y):
        
        data = zeros((x.shape[0], x.shape[1] + y.shape[1]))
        
        data[:, 0:x.shape[1]] = x
        if y.shape[1] == 1:
            data[:, x.shape[1]+y.shape[1]-1] = y.reshape(y.shape[0])
        else:
            data[:, x.shape[1]:x.shape[1]+y.shape[1]] = y

        shuffle(data)  # shuffling

        x = data[:, 0:x.shape[1]]
        if y.shape[1] == 1:
            y = data[:, x.shape[1]+y.shape[1]-1].reshape(y.shape[0], 1)
        else:
            y = data[:, x.shape[1]:x.shape[1]+y.shape[1]]
        
        return x, y