# Samples
## Convolution Neural Network for MNIST ("mnist_cnn_weight.txt")
### Summary
Achieves 99.95% accuracy on <a href="https://www.kaggle.com/c/digit-recognizer">Kaggle</a>. Trained with 1,197,766 iterations of stocastic gradient descent (batch size of one).
### Structure:
| Layer type       | Parameters                                |
|------------------|-------------------------------------------|
| input            | size: 28x28, channel: 1                   |
| convolution      | kernel: 3x3, channel: 16, padding: 0      |
| relu             |                                           |
| convolution      | kernel: 3x3, channel: 16, padding: 0      |
| relu             |                                           |
| max pooling      | kernel: 2x2                               |
| convolution      | kernel: 3x3, channel: 32, padding: 0      |
| relu             |                                           |
| convolution      | kernel: 3x3, channel: 32, padding: 0      |
| relu             |                                           |
| max pooling      | kernel: 2x2                               |
| fully connected  | size: 512                                 |
| dropout          | probability: 0.5                          |
| fully connected  | size: 256                                 |
| dropout          | probability: 0.5                          |
| softmax          | size: 10                                  |

### Hyperparameters:
| Parameter               | Value             |
|-------------------------|-------------------|
| Initial Training Rate   | 0.01              |
| Halving Rate            | 60,000 iterations |
| Regularization Cost     | 0.0001            |
| L2 Regularization       | 0.8               |
| L1 Regulatization       | 0.2               |
| Error Queue Probability | 0.0001 * size     |


## Single Layer Autoencoder for MNIST ("mnist_autoencoder_weight.txt")
Achieves a loss of 0.816 using mean squared error. Trained with 4,175,335 iterations of stocastic gradient descent (batch size of one).

### Structure:
| Layer type       | Parameters                                |
|------------------|-------------------------------------------|
| input            | size: 784                                 |
| fully connected  | size: 500                                 |
| output           | size: 784                                 |

### Hyperparameters:
| Parameter               | Value             |
|-------------------------|-------------------|
| Initial Training Rate   | 0.1               |
| Halving Rate            | 60,000 iterations |
| Regularization Cost     | 0.0001            |
| L2 Regularization       | 0.8               |
| L1 Regulatization       | 0.2               |

## Visualization:
<img src="https://raw.githubusercontent.com/jeffrey-xiao/Machine-Learning-Library/master/img/autoencoder_visualization.png" height="500"/>

## Stacked Autoencoder for MNIST ("mnist_stacked_autoencoder_weights.txt")
Achieves a loss of 26.91 using mean squared error.

### Structure:
| Layer type       | Parameters                                |
|------------------|-------------------------------------------|
| input            | size: 784                                 |
| fully connected  | size: 500                                 |
| fully connected  | size: 100                                 |
| fully connected  | size: 20                                  |
| fully connected  | size: 10                                  |
| fully connected  | size: 5                                   |
| fully connected  | size: 2                                   |
| fully connected  | size: 5                                   |
| fully connected  | size: 10                                  |
| fully connected  | size: 20                                  |
| fully connected  | size: 100                                 |
| fully connected  | size: 500                                 |
| fully connected  | size: 784                                 |

### Hyperparameters:
| Parameter               | Value             |
|-------------------------|-------------------|
| Initial Training Rate   | 0.1               |
| Halving Rate            | 60,000 iterations |
| Regularization Cost     | 0.0001            |
| L2 Regularization       | 0.8               |
| L1 Regulatization       | 0.2               |

## Visualization:
<img src="https://raw.githubusercontent.com/jeffrey-xiao/Machine-Learning-Library/master/img/layer1.png" height="500"/>
<img src="https://raw.githubusercontent.com/jeffrey-xiao/Machine-Learning-Library/master/img/layer2.png" height="500"/>
<img src="https://raw.githubusercontent.com/jeffrey-xiao/Machine-Learning-Library/master/img/layer3.png" height="500"/>
<img src="https://raw.githubusercontent.com/jeffrey-xiao/Machine-Learning-Library/master/img/layer4.png" height="500"/>
<img src="https://raw.githubusercontent.com/jeffrey-xiao/Machine-Learning-Library/master/img/layer5.png" height="500"/>
<img src="https://raw.githubusercontent.com/jeffrey-xiao/Machine-Learning-Library/master/img/mnist_2d_distribution.png" height="500"/>
