# Samples
## Convolution Neural Network for MNIST ("mnist_weight.txt")
### Summary
Achieves 99.95% accuracy on <a href="https://www.kaggle.com/c/digit-recognizer">Kaggle</a>.
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
