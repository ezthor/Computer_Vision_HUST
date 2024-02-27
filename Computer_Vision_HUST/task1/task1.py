import pandas as pd
import numpy as np
import torch

# setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
epochs = 100
learning_rate = 0.0005
# using softmax
network_type = "softmax"
# using which optimizer
optimizer_type = "Adam"
# using which loss function
loss_fn_type = "CrossEntropyLoss"
# print result
noisy = False
# print train info
print_info = True

# read dataset
dataset = pd.read_csv("./dataset.csv")

# split dataset
train_set = dataset.sample(frac=0.9, random_state=0)
test_set = dataset.drop(train_set.index)

# convert dataset to numpy
train_set = train_set.to_numpy()
test_set = test_set.to_numpy()

# construct dataloader
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=400, shuffle=True)


# print dataloader


# construct network, this dataset has 2 features and 1 label
class FNN1(torch.nn.Module):
    def __init__(self):
        super(FNN1, self).__init__()
        self.fc1 = torch.nn.Linear(2, 8)
        self.fc2 = torch.nn.Linear(8, 16)
        self.fc3 = torch.nn.Linear(16, 4)


    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        if network_type == "softmax":
            return torch.nn.functional.softmax(x, dim=-1)
        else:
            return x


# construct model
model = FNN1().to(device)
print("model construct done!")

# construct optimizer
if optimizer_type == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print("optimizer construct done!")
# construct loss function ,try more loss function
if loss_fn_type == "MSELoss":
    loss_fn = torch.nn.MSELoss(reduction="mean")
else:
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
print("loss function construct done!")

# get train size and test size
train_size = len(train_set)
test_size = len(test_set)


# train function
def train():
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            # here x is from 0 to -2, y is the last column
            # my label have 4 classes, so y should be a tensor with 4 elements, each element is 0 or 1,if label is 3,the third element is 1,others are 0
            x= data[:, :-1]
            #y should be a tensor with 4 elements, each element is 0 or 1,if label is 3,the third element is 1,others are 0
            y = data[:, -1]
            x = x.to(torch.float32)
            x = x.to(device)
            y_pred = model(x)
            # convert y to one-hot probablity to fit softmax
            if network_type == "softmax":
                y = torch.nn.functional.one_hot(y.to(torch.int64)-1, num_classes=4).to(torch.float32)
            y=y.to(device)
            #print(y_pred)
            #print(y)
            loss=loss_fn(y_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # output loss each epoch
        if print_info:
            print("epoch: {}/{},loss: {}".format(epoch,epochs,  loss.item()))
    # save model
    torch.save(model, "./model_{}_{}_{}.pth".format(network_type, optimizer_type, loss_fn_type))


# test function
def test():
    # load model
    #model = torch.load("./model_{}_{}_{}.pth".format(network_type, optimizer_type, loss_fn_type))
    with torch.no_grad():
        correct = 0
        for i, data in enumerate(test_loader):
            x = data[:, :-1]
            y = data[:, -1]
            x = x.to(torch.float32)
            x = x.to(device)
            y_pred = model(x)
            # convert y_pred from tensor to label and each +1
            y_pred = torch.argmax(y_pred, dim=-1) + 1
            y = y.to(device)
            if noisy:
                print("y_pred: {}, y: {}".format(y_pred, y))
            # calculate accuracy
            correct += (y_pred == y).sum().item()

        print("accuracy: {}".format(correct / test_size))


# train and test
if __name__ == "__main__":
    train()
    test()
