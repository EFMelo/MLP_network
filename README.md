# MLP Neural Network

> Construction of a Multilayer Perceptron Network without the use of frameworks such as Keras or PyTorch.

A MLP is featured by connected neurons arranged in many layers that form an architeture. The input layer receives the problem data, in the hidden layer, more computational capacity is allocated, and the output layer provides the results obtained. The Figure below illustrates an MLP architecture.

<p align="center">
  <img width="273" height="208" src="https://i.imgur.com/WmAVtg5.png">
</p>


### Example of Use

**MLP Configuration**

- `input_shape`: A tuple or list containing the number of samples and the number of attributes must be informed.
- `layers`: A tuple or list containing the number of neurons in each layer.
- `activations`: A tuple or list containing the activation functions in each layer. The activation functions are: _'linear'_, _'sigmoid'_, _'tanh'_ and _'relu'_
- `initializer`: A string informing the type of initialization, which can be _'glorot'_ or _'he'_. The _'glorot'_ value is the default.

```python
model = MLP(input_shape=(10000, 380), layers=(50, 50, 20, 1), activations=('relu', 'relu', 'relu', 'sigmoid'), initializer='he')
```

**MLP Training**

To train MLP, input `x`, output `y` and the number of `epochs` must be informed.

```python
history = model.run(x_train, y_train, epochs=150)
```

Some optional parameters can be informed.

- `batch_size`: The default value is _None_.
- `lr`: The default learning rate value is 0.001.
- `optimizer`: The default value of the optimizer is _'adam'_. Other values that can be entered are: _'sgd'_, _'sgd_momentum'_, _'adagrad'_ and _'rmsprop'_.
- `rho1` and `rho2`: First and second moment, respectively. Where their default values are 0.9 and 0.999. The parameter _rho2_ is used only with _adam_. 
- `shuffle`: Shuffle the data. The value _True_ is the default.

```python
history = model.run(x_train, y_train, epochs=150, batch_size=32, lr=1e-4, optimizer='sgd_momentum', shuffle=False)
```

### Datasets

**Breast Cancer**

Dataset with `519 training` samples and `50 test` samples.

Outputs:
- `0`: No breast cancer.
- `1`: With breast cancer.

**Wine Quality**

Dataset with `4500 training` samples and `398 test` samples.

Output:
- Wine quality (score between `0 and 10`)


### Some Results of MLP

**Breast Cancer**

Using the dataset:

```python
from datasets import BreastCancer
x_train, y_train, x_test, y_test, _ = BreastCancer.load_data(pp='mms')
```

The figure below compares the `MLP prevision` (left) and `Actual value` (right) for 9 values.

<p align="center">
  <img width="268" height="297" src="https://i.imgur.com/svaBucn.png">
</p>


**Wine Quality**

Using the dataset:

```python
from datasets import WineQuality
x_train, y_train, x_test, y_test, norm = WineQuality.load_data(pp='std')
```

The figure below compares the `MLP prevision` (left) and `Actual value` (right). In the center are rounded MLP values.

<p align="center">
  <img width="481" height="300" src="https://i.imgur.com/2LydV3n.png">
</p>


### Contact

[linkedin.com/in/edvaldo-melo/](https://www.linkedin.com/in/edvaldo-melo/)

emeloppgi@gmail.com

[github.com/EFMelo](https://github.com/EFMelo)