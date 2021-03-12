# Useful links
* [Writeup](https://mcgill-my.sharepoint.com/:w:/g/personal/ding_ma_mail_mcgill_ca/EWxUNSKCE5ZNsm0SmyEly9UBCVJkxfkar4Hx5oq8EvcPuA?e=RGdVWb)
* [Prof code repo](https://github.com/mravanba/comp551-notebooks)
* adding `#%%` to a line in VS Code will make that file into a Jupyter Notebook

# Experiments to run
1. > First of all, create three different models: (1) an MLP with no hidden layers, i.e., it directly maps the inputs to outputs, (2) an MLP with a single hidden layer having 128 units and ReLU activations, (3) an MLP with 2 hidden layers each having 128 units with ReLU activations. It should be noted that since we want to perform classification, all of these models should have a softmax layer at the end. After training, compare the test accuracy of these three models on the MNIST dataset. Comment on how non-linearity and network depth effects the accuracy. Are the results that you obtain expected?

1. > Take the last model above, the one with 2 hidden layers, and create two different copies of it in which the activations are now sigmoid and tanh. After training these two models compare their test accuracies with model having ReLU activations. Comment on the performances of these models: which one is better and why? Are certain activations better than others? If the results are not as you expected, what could be the reason?

1. > Create an MLP with 2 hidden layers each having 128 units with ReLU activations as above. However, this time, add L2 regularization (weight decay) to the cost and train the MLP in this way. How does this affect the accuracy?

1. > Create an MLP with 2 hidden layers each having 128 units with ReLU activations as above. However, this time, train it with unnormalized images. How does this affect the accuracy?

1. > You can report your findings either in the form of a table or a plot in the write-up. However, include in your colab notebooks the plots of the test and train performance of the MLPs as a function of training epochs. This will allow you to see how much the network should be trained before it starts to overfit to the training data.

1. Investigate on number of units in hidden layer
1. Effects of different regularization on final performance
1. Train on $10^k$ images where $k \in \{0,1,2,3,4\}$
1. Effect on data augmentation / dropout / number of hidden layers
1. Plot Model loss and train/test accuracy over Epoch

# Test Results

