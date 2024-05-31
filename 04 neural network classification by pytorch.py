# A classification problem involves predicting whether something is one thing or another.
# there are 3 types of classification 
# Binary classification	Target can be one of two options, e.g. yes or no	Predict whether or not someone has heart disease based on their health parameters.
# Multi-class classification	Target can be one of more than two options	Decide whether a photo of is of food, a person or a dog.
# Multi-label classification	Target can be assigned more than one option	Predict what categories should be assigned to a Wikipedia article (e.g. mathematics, science & philosohpy).

# neural network
# input layer -> hidden layers -> output layer

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from plots_to_help import plot_decision_boundary
from pathlib import Path
nsamples = 1000

X,y = make_circles(n_samples=nsamples,noise=0.05,random_state=42)
print(len(X),len(y))

print(X[:5],y[:5])
circles = pd.DataFrame({"X1": X[:, 0],
    "X2": X[:, 1],
    "label": y
})
print(circles.head(10))

plt.scatter(x=X[:, 0], 
            y=X[:, 1], 
            c=y,
            cmap=plt.cm.RdYlBu)

# plt.show()

# not the data we are working with is a refered as toy dataset it is small but good enough to learn 


# check input output shapes
print(X.shape,y.shape)

# data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# split data into training and test set randomly

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2,random_state=42)  #0.2 refers to 20% percentage will be converted to test 

print(len(Xtrain),len(ytrain),len(ytest),len(Xtest))

# lets build a model
# 1. set up device (use gpu if there is ) 
# 2. construct a model 
# 3. define a loss function and optimizer
# 4. make a training loop 

# device name 
print(torch.cuda.get_device_name())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# model 
# subclass nn.Module
# create nn.Linear layerthat are capable of handling the shape of out data 
# define a forward function 
# make an instance of the model and evaluate and send it to target device

class circlemodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(in_features=2,out_features=16)
        self.l2 = nn.Linear(in_features=16,out_features=10)
        self.l3 = nn.Linear(in_features=10,out_features=1)
        

        self.relu = nn.ReLU()  # relu brings the non linearity to the graph and training 
    def forward(self , x):
        return self.l3(self.relu(self.l2(self.relu(self.l1(x)))))
    
model0 = circlemodel().to(device=device)
print(model0.parameters())
print(model0.state_dict())


#  just another way to make a model using the nn.sequential 
# model1 = nn.Sequential(
#     nn.Linear(in_features=2,out_features=8),
#     nn.Linear(in_features=8,out_features=1)
# ).to(device)


# print(model1)


# loss function and optimizer 
# for regression mean absolute error or mean squared error 
# for classification we gonna use binary cross enthropy or categorical crossentropy

# loss function measrures how wrong the predictions are 

# for optimizer we can use SGD or adam thet are the most common ones

lossfn = nn.BCELoss
lossfn = nn.BCEWithLogitsLoss() # sigmoid activation function built in  this is basciaclly nn.sigmoid() + nn.bceloss()

optimizer = torch.optim.SGD(params=model0.parameters(),lr=0.01)


# out of 100 examples at what percentage does our model predicts right

def accuracy(ytrue,ypred):
    # number of preds that are true 
    correct = torch.eq(ytrue,ypred).sum().item()
    acc = (correct/len(ypred)) *100
    return acc


# train a model (step wise )
# forward pass
# calculate the loss
# optimizer zero grad
# loss backward
# optimizer step


# what is a logit ?
# In simple terms, a "logit" in machine learning represents the likelihood or probability of a certain event happening. 
# It's a numerical value that helps us understand how confident a machine learning model is 
# about a particular prediction. Logits are often used as a step in the process of converting these likelihoods into actual predictions, 
# typically by applying a function like the sigmoid function to map them to 
# a probability between 0 and 1.

# going from raw logits -> prediction probabilities -> prediction labels 
# our model is gonna give the ouputs in the form of logits 
print(model0)
# we can convert these logits into prediction probabilities by passing them to some kind of activation function 
# (nn.Sigmoid for binary cross classification and nn.softmax for multicalss classification).

# convert these prediction probabilies to prediction labels by either rounding them(binary) or tabking the argmax()(multiclass)

 
torch.manual_seed(42)
epochs = 10001

# Put all data on target device
Xtrain, ytrain = Xtrain.to(device), ytrain.to(device)
Xtest, ytest = Xtest.to(device), ytest.to(device)

def runmode():
    for epoch in range(epochs):
        # 1. Forward pass
        y_logits = model0(Xtrain).squeeze()
        ypred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels
        
        # 2. Calculate loss and accuracy
        loss = lossfn(y_logits, ytrain) # BCEWithLogitsLoss calculates loss using logits
        acc = accuracy(ytrue=ytrain, 
                        ypred=ypred)
        
        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        ### Testing
        model0.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = model0(Xtest).squeeze()              # logitts into activation function into round for binary class classification
            test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
            # 2. Calcuate loss and accuracy
            test_loss = lossfn(test_logits, ytest)
            test_acc = accuracy(ytrue=ytest,
                                    ypred=test_pred)

        # Print out what's happening
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

runmode()

# plt.title("Test")
# plot_decision_boundary(model0, Xtest, ytest) 
# plt.show()


MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "04_binary_class_classification.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)

loaded_model_1 = circlemodel()

# Load model state dict 
hehe = loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))
print(hehe)
