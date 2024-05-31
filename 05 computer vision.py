# Computer vision is the art of teaching a computer to see.
# For example, it could involve building a model to classify whether a photo is of a cat or a dog (binary classification).
# Or whether a photo is of a cat, dog or chicken (multi-class classification).
# Or identifying where a car appears in a video frame (object detection).
# Or figuring out where different objects in an image can be separated (panoptic segmentation).

# Import PyTorch
import torch
from torch import nn

# Import torchvision 
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader
# Import matplotlib for visualization
import matplotlib.pyplot as plt


from timeit import default_timer as timer
from tqdm import tqdm
from pathlib import Path
# print(torchvision.__version__,torch.__version__)


train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False, # get test data
    download=True,
    transform=ToTensor()
)


image , label = train_data[0]

# print(image,label)

# labels are 9 as e have 9 different pieces of clothing 

print(image.shape)

# [1,28,28] as 1 color channel, 28 height , 28 width

print(len(test_data.targets),len(test_data.data),len(train_data.targets),len(train_data.data))


# classes that we have in the data 
classes = train_data.classes
print(classes)

# there would be 10 classes as we had 9 labels

plt.imshow(image.squeeze(),cmap="gray")
plt.title(classes[label])
plt.axis(False)
# plt.show()


torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(classes[label])
    plt.axis(False)

# plt.show()



# now we convert all these images into batches of 32 therefore 60000/32 images per batch 
# as we will not be able to keep all the 60000 images into the memory all at once
# and doing this we train our neural netowrk more than than usual i.e. pdrage its gradients per epoch



train_dataloader = DataLoader(dataset=train_data,batch_size=32,shuffle=True)
test_dataloader = DataLoader(dataset=test_data,batch_size=32,shuffle=False)

print(len(train_dataloader),len(test_dataloader))


train_features ,train_labels = next(iter(train_dataloader))
print(train_features.shape,train_labels.shape)


# make a model 0
flatten_model = nn.Flatten()
# flatten converts [color,height,width] -> [color,height*width]

x = train_features[0]
print(x.shape)
op = flatten_model(x)
print(op.shape)


class fashion_mnist(nn.Module):
    def __init__(self,input_shape: int ,hidden_units: int,output_shape: int) :
        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,out_features=hidden_units),
            nn.Linear(in_features=hidden_units,out_features=output_shape)
        )

    def forward(self,x):
        return self.layer_stack(x)
    

torch.manual_seed(42)

model0 = fashion_mnist(input_shape=784 # this is 28*28
                       ,hidden_units=10,output_shape=len(classes))

print(model0)

x = torch.rand([1,1,28,28])
print(model0(x).shape)
 

# as we have multiclass data 
lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model0.parameters(),lr=0.1)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc



def time(start:float,end:float,device:torch.device = None):
    totaltime = end-start
    print(f"time on {device} : {totaltime} seconds")
    return totaltime

# this is how we are gonna calculate time for how much time this is gonna take
# start = timer()
# end = timer()
# time(start=start,end=end,device="cpu")

print(enumerate(train_dataloader))

# torch.manual_seed(42)
# starttime = timer()

# epochs = 3

# for epoch in tqdm(range(epochs)):

#     print("epoch : ",epoch )

#     trainloss = 0 
    
#     for batch , (X,y) in enumerate(train_dataloader):
#         # train mode 
#         model0.train()
        
#         #forward pass
#         ypred = model0(X)

#         # loss function 
#         loss = lossfn(ypred,y)
#         trainloss+=loss
        
#         # optimizer zero grad 
#         optimizer.zero_grad()

#         # back propagation
#         loss.backward()

#         #optimizer step()
#         optimizer.step()

#         if batch % 400 == 0:
#             print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

#     # Divide total train loss by length of train dataloader (average loss per batch per epoch)
#     trainloss /= len(train_dataloader)
    
#     ### Testing
#     # Setup variables for accumulatively adding up loss and accuracy 
#     test_loss, test_acc = 0, 0 
#     model0.eval()
#     with torch.inference_mode():
#         for X, y in test_dataloader:
#             # 1. Forward pass
#             test_pred = model0(X)
        
#             # 2. Calculate loss (accumatively)
#             test_loss += lossfn(test_pred, y) # accumulatively add up the loss per epoch

#             # 3. Calculate accuracy (preds need to be same as y_true)
#             test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        
#         # Calculations on test metrics need to happen inside torch.inference_mode()
#         # Divide total test loss by length of test dataloader (per batch)
#         test_loss /= len(test_dataloader)

#         # Divide total accuracy by length of test dataloader (per batch)
#         test_acc /= len(test_dataloader)

#     ## Print out what's happening
#     print(f"\nTrain loss: {trainloss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

   




# endtime = timer()

# time(start=starttime,end=endtime,device="cpu")





# # MODEL_PATH = Path("models")
# # MODEL_PATH.mkdir(parents=True, # create parent directories if needed
# #                  exist_ok=True # if models directory already exists, don't error
# # )

# # # # Create model save path
# # MODEL_NAME = "05_model0_for_computer_vision.pth"
# # MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# # # # Save the model state dict
# # # print(f"Saving model to: {MODEL_SAVE_PATH}")
# # # torch.save(obj=model0.state_dict(), # only saving the state_dict() only saves the learned parameters
# # #            f=MODEL_SAVE_PATH)
# # # Create a new instance of FashionMNISTModelV2 (the same class as our saved state_dict())
# # # Note: loading model will error if the shapes here aren't the same as the saved version
# # model = model0() 

# # # Load in the saved state_dict()
# # model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))


# def evaluate(model:torch.nn.Module,data_loader:torch.utils.data.DataLoader,lossfn,accuracy):
#     loss,acc= 0 , 0
#     model.eval()
#     for X,y in data_loader:
#         ypreds_test = model(X)

#         loss += lossfn(ypreds_test,y)

#         acc+= accuracy(y_true=y, 
#                                 y_pred=ypreds_test.argmax(dim=1))


#     loss /= len(data_loader)
#     acc /= len(data_loader)

#     print(f"loss : {loss}\n acc : {acc}\n model : {model}")



# evaluate(model0,test_dataloader,lossfn=lossfn,accuracy=accuracy_fn)


# this was model 0 with linearity

# but with non linearity we are gonna overfitting our model and adding more complexity than it is already required \
# so therefore no non linear model practice 

# we are gonna go to cnn 
# convolutional neural network
# A convolutional neural network (CNN) is a type of artificial neural network that is used for image recognition and processing.
# CNNs are also known as ConvNets.

# using 3 types of layers 
# 1. convolutional layer
# 2. relu to add non linearity
# 3. max-polling to supress a get the bigger feature 
# bascically these extract the features out of an image so that this can be helped to understand the image 

# linear and non linear layers are only used to find the patterns in these layers where as cnn are used to get the 
# features out of an image

# convolutional layer is present in conv2d , 2d is for 2 dimensional data

class model_using_cnn(nn.Module):
    def __init__(self,in_features:int,hidden_features:int,out_features:int):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features,out_channels=hidden_features,kernel_size=3,stride=1,padding=1),
            #kernel size to 3 that is we take 3X3 bloc of main image and make it into 1 feature
            # strinde is how many pixels we jump while traversing
            # padding is the extra later of 0 and 0 all around the image so as to protect the pixels that are the border of the image 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_features,out_channels=hidden_features,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_features,out_channels=hidden_features,kernel_size=3,stride=1,padding=1),
            #kernel size to 3 that is we take 3X3 bloc of main image and make it into 1 feature
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_features,out_channels=hidden_features,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_features*7*7,out_features=out_features)
        )

    def forward(self,x:torch.tensor):
        x = self.conv_block1(x)
        # print(f"this is after 1st conv layer : {x.shape}")
        x = self.conv_block2(x)
        # print(f"this is after 2nd conv layer : {x.shape}")
        x = self.classifier(x)
        # print(f"this is after classifier : {x.shape}")
        
        return x

torch.manual_seed(42)
modelcnn = model_using_cnn(in_features=1, 
    hidden_features=10, 
    out_features=len(classes))

print(modelcnn)

# torch.manual_seed(42)
# tensor = torch.rand([1,28,28])
# print(tensor.shape)

# conv = nn.Conv2d(in_channels=1,out_channels=10,kernel_size=3,stride=1,padding=1)
# maxpool = nn.MaxPool2d(kernel_size=2)

# conv_op = conv(tensor)
# print(conv_op.shape)
# # print(conv_op) 
# max_op = maxpool(conv_op)
# print(max_op.shape)

# modelcnn(image.unsqueeze(0))

# test and training the model

lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=modelcnn.parameters(),lr=0.1)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device = torch.device
               ):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        
        X,y = X.to(device)  ,y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device:torch.device
              ):
    test_loss, test_acc = 0, 0
    model.to(device)
    

    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            X,y = X.to(device),y.to(device)
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")



# torch.manual_seed(42)
# starttime = timer()

# # Train and test model 
# epochs = 5
# for epoch in tqdm(range(epochs)):
#     print(f"\nEpoch: {epoch}\n---------")
#     train_step(data_loader=train_dataloader, 
#         model=modelcnn, 
#         loss_fn=lossfn,
#         optimizer=optimizer,
#         accuracy_fn=accuracy_fn,  
#         device=device
#     )
#     test_step(data_loader=test_dataloader,
#         model=modelcnn,
#         loss_fn=lossfn,
#         accuracy_fn=accuracy_fn,
#         device=device
        
#     )

# stoptime = timer()
# time(start=starttime,end=stoptime)


# save and load our model 

from pathlib import Path

# Create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                 exist_ok=True # if models directory already exists, don't error
)

# Create model save path
MODEL_NAME = "06_pytorch_computer_vision_model_based_CNN.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
# print(f"Saving model to: {MODEL_SAVE_PATH}")
# torch.save(obj=modelcnn.state_dict(), # only saving the state_dict() only saves the learned parameters
#            f=MODEL_SAVE_PATH)


model = model_using_cnn(in_features=1,hidden_features=10,out_features=len(classes))

# Load in the saved state_dict
model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Set the model in evaluation mode
# print(model.state_dict())
print(model)




class_names = classes

def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(sample, dim=0) # Add an extra dimension and send sample to device

            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)

            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())
            
    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)


import random
random.seed(42)
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

# View the first test sample shape and label
print(f"Test sample image shape: {test_samples[0].shape}\nTest sample label: {test_labels[0]} ({class_names[test_labels[0]]})")


# Make predictions on test samples with model 2
pred_probs= make_predictions(model=model, 
                             data=test_samples)

# View first two prediction probabilities list
pred_probs[:2]

# Turn the prediction probabilities into prediction labels by taking the argmax()
pred_classes = pred_probs.argmax(dim=1)
pred_classes

plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
  # Create a subplot
  plt.subplot(nrows, ncols, i+1)

  # Plot the target image
  plt.imshow(sample.squeeze(), cmap="gray")

  # Find the prediction label (in text form, e.g. "Sandal")
  pred_label = class_names[pred_classes[i]]

  # Get the truth label (in text form, e.g. "T-shirt")
  truth_label = class_names[test_labels[i]] 

  # Create the title text of the plot
  title_text = f"Pred: {pred_label} | Truth: {truth_label}"
  
  # Check for equality and change title colour accordingly
  if pred_label == truth_label:
      plt.title(title_text, fontsize=10, c="g") # green text if correct
  else:
      plt.title(title_text, fontsize=10, c="r") # red text if wrong
  plt.axis(False);



plt .show()