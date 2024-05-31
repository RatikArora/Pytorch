import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

weight = 0.9
bias = 0.6

start = 0
end = 1
step = 0.02

tensor = torch.arange(start,end,step).unsqueeze(dim=1)

y = weight*tensor +bias
# print(len(y))
X = tensor
train_split = int(0.7*len(X))
X_train , y_train = X[:train_split] , y[:train_split]
X_test , y_test = X[train_split:] , y[train_split:]


# print(len(X_train),len(X_test),len(y_train),len(y_test))

def plot_pred(train_data = X_train,train_lable = y_train,test_data=X_test,test_lable=y_test,prediction = None):
    # to changethe window sizes  
    plt.figure(figsize=(10,7))
    plt.scatter(train_data,train_lable,c='b',s=4,label="training data")
    plt.scatter(test_data,test_lable,c='g',s=4,label="testing data")

    if prediction is not None:
        plt.scatter(test_data,prediction,c='r',s=4,label="predictions")

    # legend creates the scale on the top corner
    plt.legend(prop={"size":10 })


plot_pred()
plt.title("NO Prediction")
# plt.show()  


class linearregressionmodel(nn.Module):
    def __init__(self):
        super().__init__()  # usig the inner made linear regression model bu the torch.nn
        self.linear_layer = nn.Linear(in_features=1, out_features=1) # in and out refers to the number for inputs for number of output that the model is gonna give here y is derived from 1 value of x 

    def forward(self, x : torch.tensor) -> torch.tensor:
        return self.linear_layer(x)
    

torch.manual_seed(42)
model1= linearregressionmodel()
print(model1.state_dict())

with torch.inference_mode():
    ypreds = model1(X_test)
    print(ypreds)

plt.title("First Prediction")
plot_pred(prediction=ypreds.cpu())


# define a loss function and optimizer
lossfn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model1.parameters(), lr=0.005) # optimize newly created model's parameters 
# now make a traiing loop 

epochs = 300

for epoch in range(epochs):

    model1.train()
    ypreds = model1(X_train)
    loss = lossfn(ypreds,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model1.eval()
    if epoch % 10 == 0:
        print(model1.state_dict())


with torch.inference_mode():
    ypreds = model1(X_test)

print(ypreds)

plot_pred(prediction=ypreds.cpu())
plt.title("POST Training")
plt.show()


print(f"the self declared values were {weight} , {bias}")
print(f"the derived values are {model1.state_dict()}")


# to save the model we can use the torch.save method 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "03_linear_regression.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model1.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)

loaded_model_1 = linearregressionmodel()

# Load model state dict 
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))