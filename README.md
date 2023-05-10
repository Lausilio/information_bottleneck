# Information Bottleneck Theory for Convolutional Neural Networks

This repository contains the implementation of the Information Bottleneck Theory applied to Convolutional Neural Networks (CNNs), inspired by the Saxe et al. paper. The Information Bottleneck Theory aims to provide a deeper understanding of the underlying principles in deep learning networks and their generalization properties. Our project expands the original work by applying the theory to CNNs and offering three different methods for computing mutual information.

## Project Overview

The primary goal of this project is to investigate the information bottleneck theory in the context of convolutional neural networks, which are widely used for various computer vision tasks. By implementing this theory for CNNs, we aim to provide insights into their internal representations and generalization properties.

To achieve this, we compute the mutual information between the input and internal representations of the network, as well as between the internal representations and output. In order to obtain accurate and robust estimates of mutual information, we utilize three different methods:

1. Binning method
2. Kernel Density Estimation (KDE) method
3. Kraskov method

By comparing the results obtained from these methods, we provide a comprehensive analysis of the information bottleneck theory for CNNs.

To see our presentation click [here](https://github.com/Lausilio/information_bottleneck/blob/master/Information%20theory.pdf)
