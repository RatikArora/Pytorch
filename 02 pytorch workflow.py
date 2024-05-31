# workflow

# get the data ready
# build or pick a trained model to suit your problem 
# fitting the model to data 
# make pridictions 
# save and load a model 
# putting it all together 


import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path


# print(torch.__version__)

# 1. data preparing and loading
# excel,images,videos,audio,dna,text 

# 1.1 get data into a numerical representaion
# 1.2 get a model and train it based on your data to find patterns in that numerical representation

#  starting with linear regression model 
# y = bx + a 
weight = 0.7 #(b)
bias = 0.3  #(a)

start = 0
end = 1
step = 0.02

tensor = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * tensor + bias

# print(y,tensor)
# print(len(y),len(tensor))
X = tensor

# making 3 datasets 
# training set
# validation set
# test set


# 1. training and test set from our data 

train_split = int(0.8*len(X))
X_train , y_train = X[:train_split] , y[:train_split]
X_test , y_test = X[train_split:] , y[train_split:]

# print(len(X_train),len(y_train),len(X_test),len(y_test))


# visulaize the data 

def plot_pred(train_data = X_train,train_lable = y_train,test_data=X_test,test_lable=y_test,prediction = None):
    # to changethe window sizes  
    plt.figure(figsize=(10,7))
    plt.scatter(train_data,train_lable,c='b',s=4,label="training data")
    plt.scatter(test_data,test_lable,c='g',s=4,label="testing data")

    if prediction is not None:
        plt.scatter(test_data,prediction,c='r',s=4,label="predictions")

    # legend creates the scale on the top corner
    plt.legend(prop={"size":10 })


# no predictions were sent to uncomment this to look at the testing and training variables graph
plot_pred()
# plt.show()


# 2. build model 

# create a linear regression model class 

class linearregressionmodel(nn.Module): 
    def __init__(self) :
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float))

    def forward(self,x:torch.tensor) -> torch.tensor: # x ix the input data 
        return self.weights*x+self.bias # linear regression formula
    


# nn.Module contains the larger building blocks (layers)
# nn.Parameter contains the smaller parameters like weights and biases (put these together to make nn.Module(s))
# forward() tells the larger blocks how to make calculations on inputs (tensors full of data) within nn.Module(s)
# torch.optim contains optimization methods on how to improve the parameters within nn.Parameter to better represent input 


torch.manual_seed(22)

model0 = linearregressionmodel()
print(model0.state_dict())


with torch.inference_mode(): 
    y_preds = model0(X_test)
    # print(y_preds)


plot_pred(prediction=y_preds)
# uncomment this to see the 1st prediction that is made by the model
# plt.show()

# now we want to make these red predictions to get close to the real ones 

# this is how you make and plot a prediction 
# now we need a training loop that will train our model 
# we can use 2 features of pytorch 
# 1st optimizer   torch.optim   
# 2nd loss function     torch.nn

# in loss functions we have many types but here we are gonna use the l1 loss method 
# l1 loss method uses the mean absolute error formula to calculate the loss 
# and for optimizer we are gonna use the SGD optimizer 
# SGD optimizer takes 2 values 
# 1st paramerters of the model 
# 2nd learning rate that is the hyperparameter that we can adjust the lower it the the bett the optimization but slower the learining rate and higher the time 


# # Create the loss function
loss_fn = nn.L1Loss() # MAE loss is same as L1Loss
# # print(loss_fn)
# # Create the optimizer
optimizer = torch.optim.SGD(params=model0.parameters(), # parameters of target model to optimize
                            lr=0.01) # learning rate (how much the optimizer should change parameters at each step, higher=more (less stable), lower=less (might take a long time))

# print(optimizer)

###########################################################################################################################################################################################################################################################
# very important steps to make a training loop 

# 1	Forward pass	The model goes through all of the training data once, performing its forward() function calculations.	model(x_train)
# 2	Calculate the loss	The model's outputs (predictions) are compared to the ground truth and evaluated to see how wrong they are.	loss = loss_fn(y_pred, y_train)
# 3	Zero gradients	The optimizers gradients are set to zero (they are accumulated by default) so they can be recalculated for the specific training step.	optimizer.zero_grad()
# 4	Perform backpropagation on the loss	Computes the gradient of the loss with respect for every model parameter to be updated (each parameter with requires_grad=True). This is known as backpropagation, hence "backwards".	loss.backward()
# 5	Update the optimizer (gradient descent)	Update the parameters with requires_grad=True with respect to the loss gradients in order to improve them.	optimizer.step()

############################################################################################################################################################################################################################################################


epochs = 400


for epoch in range(epochs):
    ### Training

    # Put model in training mode (this is the default state of a model)
    model0.train()

    # 1. Forward pass on train data using the forward() method inside 
    y_pred = model0(X_train)
    # print(y_pred)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad of the optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Progress the optimizer
    optimizer.step()

    # testing
    # printing the model paramerts based on epochs training the model 
    if epoch % 50 ==0:
        print(model0.state_dict())



print(model0.state_dict())

# set the model up for the evaluation mode 
model0.eval()

# plot preditions after training 
with torch.inference_mode():
  y_preds = model0(X_test)
plot_pred(prediction=y_preds)
# uncomment this to look that what the graph looks like when yo have done training 
# plt.show()  



# # 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# # 2. Create model save path 
MODEL_NAME = "02_pytorch_workflow_model0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)



loaded_model0 = linearregressionmodel()

# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
print("the loaded model dicts are ",loaded_model0.state_dict())


