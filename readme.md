<h1 align=center> Vanilla Neural Network Using Python </h1>
<p align="center">
<a href="https://travis-ci.org/karanvivekbhargava/vanilla-neural-network">
<img src="https://travis-ci.org/karanvivekbhargava/vanilla-neural-network.svg?branch=master">
</a>


<p align="center">
<img src="https://www.pyimagesearch.com/wp-content/uploads/2016/08/simple_neural_network_header.jpg">

Reference for image: <a href='https://www.pyimagesearch.com'>link</a>
</p>

## Project Overview

This is an implementation of an optimized neural network using python. Various optimization methods have been used in conjunction to make the network converge faster. Sigmoid non-linearities have been sandwiched between the layers. The final loss function has been chosen as the softmax layer. The neural network is very flexible and can be used with different number of layers.

The Barzilai-Borwein method has been used to solve the optimization problem.

## Dependencies

This project uses the following:
* `python 2.7`
* `numpy 1.11.0` - for matrix manupulations
* `sklearn` - to load data (this has a dependency with `scipy`)
* `matplotlib` - to visualize the loss

To install them kindly enter the following on the terminal
```
pip install numpy==1.11.0
pip install scipy
pip install sklearn
pip install matplotlib
```

## How to build demo - standard install via command-line
The code below runs the example on the iris dataset. Since its a small dataset with low dimensions, the algorithm converges quite fast.
```
git clone --recursive https://github.com/karanvivekbhargava/vanilla-neural-network.git
cd <path to repository>
python main.py
```
In order to view the graph for the residuals during training, change the False to True on `line 17` of `main.py`. This was done for the continuous integration on Travis.

## Example

The neural network class example is given below

```
# Instantiate neural network object
nn = Net()
# Pre-process data
x_train, y_train, x_valid, y_valid, x_test, y_test = readData()
# Feed data into neural network
nn.feedData(x_train, y_train, x_valid, y_valid, x_test, y_test)
# Add the layers
nn.addLayer(50)
nn.addLayer(40)
nn.addLayer(30)
nn.addLayer(20)
nn.addLayer(y_train.shape[1])
# Start the NN solver
nn.solver()
```
The `Net()` initializes the net object. After this, the data needs to be fed into the network by using the `feedData` method. We can freely add layers of all dimensions using `addLayer` method.

The `readData` function reads the iris dataset and normalizes the data so that the neural network converges faster.

<p align="center">
<img src = "UML/Activity_Diagram_v2.jpg">
</p>

## Results

Sample output:

<p align="center">
<img src = "result.png">
</p>

Terminal output:
```
Training Accuracy =  98.0952380952
Test Accuracy =  96.7741935484
```
