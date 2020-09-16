from nn import MLP
from dataset import DataSet
import matplotlib.pyplot as plt

x_train, y_train, x_test, y_test, _ = DataSet.breast_cancer(pp='mms')

# Training
in_shape = (519, 30)  # input format -> (number_of_samples, number_of_attributes)
layers = (30, 30, 1)  # Three Layers with 30-30-1 neurons
functions=('relu', 'relu', 'sigmoid')  # Activation function of each layer

epochs = 300
model = MLP(input_shape=in_shape, layers=layers, activations=functions, initializer='he')
history = model.run(x_train, y_train, epochs=epochs, batch_size=32)

# Testing
prediction = model.predict(x_test)

# Plotting the Loss
plt.plot(range(epochs), history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()