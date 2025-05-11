#!/usr/bin/env python
# coding: utf-8

# In[433]:


import numpy as np

import math
import random
import matplotlib.pyplot as plt
import pandas as pd


# In[434]:


# create a class for perceptron 


class Preceptron():
    def __init__(self):
        self._activation = 0

    def calc_activation(self,weight, prev_activation):
        self._activation = (weight*prev_activation)

    @property
    def activation(self):
        return self._activation

    @staticmethod
    def sigmoid(z):
        return 1/(1+math.e ** (-z))


    def __repr__(self):
        return str(self._activation)


class InputPerceptron(Preceptron):
    def __init__(self):
        super().__init__()

    def calc_activation(self, weight,prev_activation):
        self._activation =  prev_activation



# In[435]:


class NeuralNetwork:
    def __init__(self, learing_rate = 0.1):
        self.lr = learing_rate
        self.neuron: list[Preceptron] = [InputPerceptron()]+ [Preceptron() for _ in range(3)] 
        self.weight: list[float] =  [random.random() for _ in range(3)]

    def forward_pass(self, input):
        prev= input
        for i in range(len(self.neuron)):
            if i == 0:
                self.neuron[i].calc_activation(1, prev)
                # print(input, prev)
            else:
                self.neuron[i].calc_activation(self.weight[i-1], prev)

            prev = self.neuron[i].activation



    def gradient(self, y):
        res = []
        n = len(self.neuron)-1

        deltaC_al = 2*(self.neuron[n].activation-y)


        for i in range(len(self.weight)):
            # delat every layer has 
            # print(f"@c/@a{n}")

            temp = deltaC_al
            for j in range(i+1, len(self.weight)):
                # print(f"w{j}")
                temp = temp*self.weight[j]

            # print(f"σ'(a{i} * w{i}) * a{i} ")
            # print(f"a{i}")

            z = self.neuron[i].activation * self.weight[i]
            temp = temp * self.neuron[i].activation 
            # temp = temp * self.neuron[i].activation


            res.append(temp)
            # print("---------")

        return res



    def train_batch(self, input_data):
        # forward pass
        avg_grad = [0 for _ in range(len(self.weight))]
        for x, y in input_data:

            self.forward_pass(x)       

            new_grad = self.gradient(y)
            for i in range(len(new_grad)):
                avg_grad[i] += new_grad[i]


        n = len(input_data)

        avg_grad = list(map(lambda val: val/n, avg_grad))

        self.weight = list(map(lambda ix: ix[1] - self.lr * avg_grad[ix[0]], enumerate(self.weight)))


    def predict(self, X):
        res = []
        for x in X:
            self.forward_pass(x)
            res.append(self.neuron[-1].activation)
        return res



# In[436]:


nn = NeuralNetwork()
print(nn.weight)
nn.train_batch([[10.0,10.0]])
print(nn.weight)


# In[437]:


# creating dummy data 
train_data =[]
for i in range(100):
    x=  random.randint(-100,100)
    y = x*2
    train_data.append([x,y])

train_data.sort()
train_data = pd.DataFrame(train_data, columns=["X", "Y"])
plt.scatter(train_data["X"] ,train_data["Y"])


x_min = train_data["X"].min()
x_max = train_data["X"].max()

y_min = train_data["Y"].min()
y_max = train_data["Y"].max()



train_data["X_norm"] = train_data["X"].apply(lambda x: (x-x_min)/(x_max-x_min))
train_data["Y_norm"] = train_data["X"].apply(lambda y: (y-y_min)/(y_max-y_min))



train_data


# In[438]:


train_data_np = train_data[["X_norm", "Y_norm"]].to_numpy()



sin_nn = NeuralNetwork()
train_batch = []
for i in range(30):
    train_batch.append(train_data_np[i:i+1*10, :])

for i in range(100):
    for batch in train_batch:
        sin_nn.train_batch(batch)


# In[439]:


sin_nn.weight


# In[443]:


y_pred = sin_nn.predict(train_data["X_norm"])  # normalized prediction

plt.scatter(train_data["X_norm"], train_data["Y_norm"], label="True (Normalized)")
plt.scatter(train_data["X_norm"], y_pred, label="Predicted (Normalized)")
plt.legend()
plt.show()


# In[444]:


# Denormalize predictions
y_pred = list(map(lambda y: y * (y_max - y_min) + y_min, y_pred))

# Plot original data vs. denormalized predictions
plt.scatter(train_data["X"], train_data["Y"], label="True (Original Scale)")
plt.scatter(train_data["X"], y_pred, label="Predicted (Denormalized)")
plt.legend()
plt.show()


# In[442]:


y_pred


# In[ ]:




