# Machine Learning Architectures & Deep Learning Implementations

This is a meta-repository that includes all architectures / regularization techniques I implemented from scratch, it provde to be an important step in grasping the design choices, the inductive biases and backpropagation. 

If you asked me to summarize in a simple terms what heurisitc I have developed, my answer would be that learning comes in two stages:

1. *Early stages:* in early stages, the network tries to discover a *lower dimensional manifold* that the high-dimensional data (like images or text) actually lies on or near to, this manifold represents the intrinsic features of the data. The learned manifold is somewhat general and not dependent on the downstream task, we can see this, in something like $CNNs$ where earlier layers learn to detect edges regardless of the downstream task. It boils down to finding an embedding of the data, and we always do something on these lines in the different architectures, we use architectures that are tailored for the specific data at hand, to facilitate the extraction of such embedding efficiently by encoding our own inductive biases in dealing with data.
2. *Later stages:* The network learns some form of *soft-clustering* of data points in the learned salient representation depending on the downstream task, this is usually done through an $MLP$ that takes the learned embeddings as input.  

Now this view is oversimplified since the network is trained end-to-end and these steps are not preformed explicitly per se but rather deeply intertwined, the main counter-argument for this view is how full-finetunning preforms miles better than linear probing. 

## 📌 Table of Contents
- [Multi-Layer Perceptron](#multi-layer-perceptron)
- [Optimizers and Initialization Schemes](#optimizers-and-initialization-schemes)
- [BatchNorm and Dropout](#batchnorm-and-dropout)
- [Convolutional Neural Network](#convolutional-neural-network)
- [Recurrent Neural Network](#recurrent-neural-network)
- [Autoencoder](#autoencoder)
- [Transformer](#transformer)

---

## 📈 Architectures & Regularization techniques:

### Multi-Layer Preceptron:

- 📂 **Project Repo:** [Multi-Layer Preceptron](https://github.com/fadibenz/FullyConnectedNN-Vanilla-Numpy)
- 📝 **Description:** A modular implementation of fully connected neural networks from scratch, featuring forward and backward propagation, various layer types, and a flexible solver for training.

---

### Optimizers and Initialization Schemes:

- 📂 **Project Repo:** [Initialization Schemes](https://github.com/fadibenz/Optimization_Initialization_FullyConnedctedNN)
- 📝 **Description:**  A modular implementation of Momentum and Adam optimizers and the different initizalization schemes (Xavier, He)

---

### BatchNorm and Dropout:
- 📂 **Project Repo:** [BatchNorm and Dropout](https://github.com/fadibenz/BatchNorm-Dropout-VanillaNumpy)
- 📝 **Description:** A modular implementation of Batch Normalization and Dropout forward and backward pass, with symbolic computing of the derivatives for backpropagation (BatchNorm proved a bit tricky to work out)

---


### Convolutional Neural Network:

- 📂 **Project Repo:** [CNN](https://github.com/fadibenz/Convolution-SpatialBatchNorm-VanillaNumpy)
- 📝 **Description:** A modular implementation of convolutional networks from scratch, focusing on forward and backward passes for convolutional layers, max pooling, and spatial batch normalization, in addition to the full derivations for backpropagation.

---

### Recurrent Neural Network:

- 📂 **Project Repo:** [Recurrent Neural Network](https://github.com/fadibenz/RNNs-FromScratch-Gradient-Analysis)
- 📝 **Description:** A modular implementation of a simple Recurrent Neural Network (RNN) layer  using PyTorch and an RNN-based regression model built on top.
--- 

### Autoencoder:

- 📂 **Project Repo:** [Autoencoder](https://github.com/fadibenz/Autoencoders_FromScratch)
- 📝 **Description:**  Implementation of three types of autoencoders (Vanilla, Denoising, Masked) using synthetic and MNIST datasets.It includes implementations, training workflows, and performance evaluations using linear probing. 
---
### Transformer:

- 📂 **Project Repo:** [Transformer](https://github.com/fadibenz/Transformer-Summarization-)
- 📝 **Description:** Full Implementation of Transformer (Scaled-dot product, Multi-head Attention, transformer layers and encoder/decoder with causal and non-causal attention)

--- 
