import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# print(torch.__version__)


# introduction to tensors : A PyTorch Tensor is basically the same as a numpy array: it does not know anything about deep learning or computational graphs or gradients, and is just a generic n-dimensional array to be used for arbitrary numeric computation.
# Tensors are the basic building block of all of machine learning and deep learning.
# creating tensors

# 1. scalar
# scalar = torch.tensor(7)
# # 0 dimensions in scalar
# print(scalar.ndim)
# print(scalar)
# # to get the item back fromt he cell
# print(scalar.item())

# # 2. vector
# vector = torch.tensor([7,7])
# # vecotr is something that has both magnitude and direction
# print(vector)
# # ndim is number of dimensions
# print(vector.ndim)
# print(vector.shape)


# # 3. MATRIX
# matrix = torch.tensor([[7,8],[9,10]])
# print(matrix)
# print(matrix.ndim)
# print(matrix[0],matrix[1])
# # shape would be the 2 by 2 that means 2 arrays with 2 element each
# print(matrix.shape)


# # 4. TENSOR
# ts = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]])
# print(ts)
# print(ts.ndim)
# # shape is 1,3,3 
# print(ts.shape)
# print(ts[0][0].item,ts[0][1].item,ts[0][2].item)
# # here 1,3,3 refers to 1 is the main one list 3 is the number of elements of list and next 3 is very inner dimensions that is also 3

# ts2 = torch.tensor([[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]])
# print(ts2)
# print(ts2.shape)
# here the size would be 1,4,3
# understood now 


# 5. random tensors
# why random tensors ? 
# ans. random tensors are imp b/c the way many neiral network learn is that they start with tensors full of
# random numbers and then adjust those random numbers to better represent the data 

# start with random numbers -> look at data -> update random numbers -> look at data ->update random numbers

# # create a random tensor of size (3,4)
# rand_ts = torch.rand(3,4)
# print(rand_ts.ndim)
# print(rand_ts)

# create a random tensor of a size of a image 
# image_rand_ts = torch.rand(size=(224,224,3)) 
# height width and color of the image , colors are (r,g,b)
# print(image_rand_ts.shape,image_rand_ts.ndim)
# print(image_rand_ts)


# # 6. zeroes and ones
# zeros = torch.zeros(3,4)
# print(zeros)
# ones = torch.ones(3,4)
# print(ones)
# # default datatype is tensor float
# print(ones.dtype)

# 7. create a range of tensors and tensors-like
# one_to_ten = torch.arange(1,11)
# print(one_to_ten)
# # tensor like
# # like method creats the tensors that are like the main tensor input
# ten_zeroes = torch.zeros_like(one_to_ten)
# ten_ones = torch.ones_like(one_to_ten)
# print(ten_zeroes,ten_ones)



# tensor datatypes is one of the 3 big errors you will run into with pytorch & deep learning :
# 1. tensors not right database - use tensor.dtype
# 2. tensors not right shape - use tensor.shape
# 3. tensors not on the right device - use tensor.device as gpu doesnt support the tensor to numpy feature

# ts = torch.tensor([3,4,5],
#                 dtype=None,  # what datatype the tensor is float32 or float 16 etc.
#                 device=None, #what device is your tensor on
#                 requires_grad=False # weather or not to track gradients in tensor
#                 )

# # conversion of 32 bit tensor to 16 bit tensor
# float_16_tensor = ts.type(torch.float16)
# print(float_16_tensor)

# print(ts*float_16_tensor)

# float_32_tensor = ts.type(torch.float32)
# print(float_32_tensor)

# int_32_tensor = torch.tensor([3,6,9],dtype=torch.int32)
# print(int_32_tensor)
# print(int_32_tensor*float_32_tensor)

# ts = torch.rand(5,7)
# ts2 = torch.randint(low=0,high=100,size=[3,4],dtype=torch.int32)
# print(f"dtype {ts.dtype} , shape {ts.shape} ,device {ts.device}")
# print(ts2)

#####################################
# operations in tensor 

# 1. add
# 2. sub 
# 3. multiply
# 4. divide 
# 5. matrix multiplication
#####################################


# tensor = torch.tensor([[1,2,3],[4,5,6]])
# print(tensor+10)
# print(tensor-10)
# print(tensor*10)
# print(tensor/10)
# # inbuilt are also availiable 
# print(torch.mul(tensor,10))



# # matrix multiplication
# tensor = torch.tensor([1,2,3])
# print(tensor.shape) 
# print(torch.matmul(tensor,tensor))
# # another way of matrix multiplication
# print(tensor @ tensor)

# t1 = torch.rand(3,2)
# t2 = torch.rand(2,3)
# print(t1 @ t2) # this is possible 
# print("t1 = ",t1)
# print("t2 = ",t2)

# # print( t1 @ t1) not possible as 3X2 . 3X2 not possible
#  # this tye of error is called shape error

# rules of matrix multiplication
# 1. the inner dimensions must match
# 2. the outer dimensions give the shape or size of resultant matrix

 
# #  to fix shape os a matrix we use the transpose of a matrix 
# tensor = torch.tensor([[1,2,3],[4,5,6]])
# print(tensor)
# print(tensor.shape)
# # transpose of the matrix
# print(tensor.T)
# print(tensor.T.shape)

# # using this in matrix multiplication

# print("resultant matrix \n",tensor @ tensor.T)


##################################################################

# # find the min, max,mean,sum etc(tensor aggregation)
# x = torch.arange(0,101,10)
# print(x.min())
# print(torch.min(x))

# print(x.max())
# print(torch.max(x))

# # the mean only works on float and complex numbers , not on int or long
# print(x.dtype)
# print(x.type(torch.float32).mean())
# print(torch.mean(x.type(torch.float32)))

# print(x.sum())
# print(torch.sum(x))




# finding positional min and max

# y = torch.arange(100,1,-7)
# print(y)
# print(y.argmax())  # max postion  
# print(y[y.argmax()])

# print(y.argmin()) # min position
# print(y[y.argmin()])

#################################################################

# reshaping stacking squeezing and unsqueezing tensors

# reshaping - reshape an uinput tensor to a defined shape
# view - return a view of an input tensors of certain shape but keep the same memory as the original tensor
# stacking - combine multiple tensors on top of each other (vstack) or side by side (hstack)
# squeeze - removes all 1 dimensions from a tensor
# unsqueeze - add a 1 dimension to a target tensor
# premute - return a view of the input with dimensions premuted in a certain way



# reshaping
# x = torch.arange(1,11)
# print(x,x.shape)

# y = x.reshape(1,10)
# print(y,y.shape)
# y = x.reshape(2,5)
# print(y,y.shape)
# y = x.reshape(10,1)
# print(y,y.shape)
# y = x.reshape(5,2)
# print(y,y.shape)

 
# view
# a view shares a original memory of a tensor so if you make changes in that copy it will change the original as well
# x = torch.arange(1,11)
# print(x,x.shape)
# z = x.view(10)
# z[0] = 7 
# print(z,x)

# stack
# x = torch.arange(1,11)
# print(x,x.shape)
# tstacked = torch.stack([x,x,x,x],dim=0)  # vtstacked
# print(tstacked)
# tstacked = torch.stack([x,x,x,x],dim=1) # hstacked
# print(tstacked)


# squeeze
# x = torch.arange(1,11)
# print(x,x.shape)
# y = x.reshape(1,10)
# print(y,y.shape)
# z = y.squeeze()
# print(z,z.shape)

# # unsqueeze
# z = z.unsqueeze(dim=1)
# print(z,z.shape)

# x = torch.arange(1,101)
# print(x,x.shape)
# y = x.reshape(4,25)
# print(y,y.shape)
# z = y.unsqueeze(dim=1)
# print(z,z.shape)
# z = y.unsqueeze(dim=0)
# print(z,z.shape)


# # permute
# x = torch.randn(2,3,5)
# print(x,x.shape)
# y = torch.rand(2,3,4,5,3,2,3,2,3,100)
# print(y.shape)
# print(y.permute(9,8,5,6,7,4,3,2,1,0).shape)
# # changng the positions of dimensions by their sequence 
# # print(x.permute(2,0,1))

###############################################################

# # indexing in tensors

# x = torch.arange(1,10)
# x = x.reshape(1,3,3)
# print(x,x.shape)
# print(x[0])
# print(x[0][0])
# print(x[0][2][2])
# print(x[:,2,2])

###########################################################

# numpy with pytorch 

# conversion of numpy into tensors and vice versa
# array = np.arange(1.,9.)
# print(array)
# tensor = torch.from_numpy(array)                    #.type(torch.float32) to convert back to default data type of tensor
# print(tensor)


# conversion of tensor to numpy 
# tensor = torch.arange(1,10).type(torch.float32)
# array = tensor.numpy()
# print(tensor,array)
# print(array.dtype,tensor.dtype)

##############################################3

# reproducability

# random seed
# Sets the seed for generating random numbers to a non-deterministic random number.
# flvaouring the random process
# rs = 100
# torch.manual_seed(rs) 
# t1 = torch.rand(2,3)

# torch.manual_seed(rs) 
# t2 = torch.rand(2,3)
# print(t1==t2)

