# Regression Analysis and Neural Networks


## Purpose
Project related to the Statistical Methods For Machine Learning course for the Computer Science Master at Università degli Studi di Milano.

This repository is intended to keep trace of the project development, storing both the code and the project report.

## Overview
In this project I dealt with different regression problems by using neural networks as training models.

Several experiments were made to test up regression problems with neural networks, such as using appropriate preprocessing techniques, varying the networks' topologies (and so their parameters) and the activation functions used by their layers, testing different training algorithms and loss functions.
A validation loop over some validation epochs was used to obtain a good training algorithm's parameter (learning rate) for the subsequent training loop, and external cross-validation was used to obtain the model's accuracy as the mean of the different test errors computed as mean absolute error.

I tested both the linear regression and the logistic regression approachs, depending on the dataset used. I used TensorFlow's low level API to implement the models. For each test I used the cross-validation technique to compute the accuracy of the model in terms of mean error.
