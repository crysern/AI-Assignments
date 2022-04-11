# .retina() Chosen from SciKit Image Library

![image](https://git.arts.ac.uk/storage/user/189/files/a8dc9a80-a300-11ec-88cf-ec2ffbd253bf)


**Experimentation with colour channels:**
**Blue, Green, Red and RGB**

![image](https://git.arts.ac.uk/storage/user/189/files/af6b1200-a300-11ec-841e-69001c05dc39), ![image](https://git.arts.ac.uk/storage/user/189/files/c3af0f00-a300-11ec-8c03-7186f81f4015),![image](https://git.arts.ac.uk/storage/user/189/files/cc074a00-a300-11ec-8942-5588d979c9bf), ![image](https://git.arts.ac.uk/storage/user/189/files/d3c6ee80-a300-11ec-8313-98b8bc472892)



**Experimentation with noise:**

![image](https://git.arts.ac.uk/storage/user/189/files/d9bccf80-a300-11ec-8375-f8c0a7c20807),![image](https://git.arts.ac.uk/storage/user/189/files/e17c7400-a300-11ec-902d-e008c10560ca)




Extend the model in this notebook into one which maps (X,Y) -> (R,G,B).

![image](https://git.arts.ac.uk/storage/user/189/files/8c3e6380-a2fc-11ec-8359-650dc9916156)


Add at least 2 more layers to the network.

Default layers:


![image](https://git.arts.ac.uk/storage/user/189/files/9d877000-a2fc-11ec-8c1c-f1d29b2e4873)



**Experiment with alternative activation functions and optimizers.**

**Experimentation 1:**

4 Dense Layers
Activation - Tahn
Optimizer - Adam

![image](https://git.arts.ac.uk/storage/user/189/files/bee85c00-a2fc-11ec-9289-182613229c28)


We see a significant difference using completely different epochs and all new activation layers. It seems to do well with thesilhouette of any given image, and with faces and outlined shapes, the results would be pretty good.


**Experimentation 2:** 


7 Dense Layers
Activation - Relu and Sigmoid
Optimizer - SGD

200 EPOCHS

![image](https://git.arts.ac.uk/storage/user/189/files/07077e80-a2fd-11ec-88d1-4278313cb468)


500 EPOCHS


![image](https://git.arts.ac.uk/storage/user/189/files/0cfd5f80-a2fd-11ec-8d3e-f432fafd4601)


As you can see from the above SGD struggled to create an image output.

**Experimentation 3:**

7 Dense Layers
Activation - Relu and Sigmoid
Optimizer - Adam

![image](https://git.arts.ac.uk/storage/user/189/files/269ea700-a2fd-11ec-8fd7-ff2d7dc491cd)



**Reflections:**

Adam is the best optimizer if one wants to train the neural network in less time and more efficiently than Adam isthe optimizer.
For sparse data, use the optimizers with a dynamic learning rate.
Min-batch gradient descent is the best optionif we want to use a gradient descent algorithm.

Referencing:
https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6


How the image we have created differs from a normal image:


The outcome of the lesson above was to create a continuous function where the pixel value is not discrete.
The image we created differs from a normal one because it comprises many interconnected processing nodes, neurons, that can learn to recognize patterns of input data. An advantage of this method could be very effective at tasks such as image recognition or classification.


