import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np

print("import success!")
# setting
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 45


# 创建自定义数据集类，用于创建图片对
# 两两之间建立图片对，用百分之十的数据集就可以创建数据量和原来大小一样的数据集
class PairsDataset(Dataset):
    def __init__(self, mnist_dataset):
        self.data = mnist_dataset
        self.pairs = self.create_pairs()

    # 创建的数据集要相同和不相同的图片对比例为1:1，决定先把0-9的图片分开，对类别i里的图片k，随机取一张和他不同的类别i里的图片j，这样就创建了一对相同的图片对，然后再随机取一张和他不同的类别k中的图片p，这样就创建了一对不同的图片对
    def create_pairs(self):
        # 先把0-9的图片分开
        data = {}
        for i in range(10):
            data[i] = []
        for img, label in self.data:
            data[label].append(img)
        # 创建相同的图片对
        pairs = []
        for i in range(10):
            for times in range(5):
                for k in range(len(data[i])):
                    # 随机取一张和他不同的类别i里的图片j
                    j = random.randint(0, len(data[i]) - 1)
                    while j == k:
                        j = random.randint(0, len(data[i]) - 1)
                    # 创建一对相同的图片对
                    pairs.append((data[i][k], data[i][j], 1))
        # 创建不同的图片对
        for i in range(10):
            for times in range(7):
                for k in range(len(data[i])):
                    # 随机取一张和他不同的类别k中的图片p
                    p = random.randint(0, 9)
                    while p == i:
                        p = random.randint(0, 9)
                    j = random.randint(0, len(data[p]) - 1)
                    # 创建一对不同的图片对
                    pairs.append((data[i][k], data[p][j], 0))
        # 打乱顺序
        random.shuffle(pairs)
        # print一下个数
        print("pairs num:", len(pairs))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1, img2, label = self.pairs[idx]
        return img1, img2, label


# 数据预处理和加载
# transform是一个转换器，]Compose是将多个转换器组合起来
transform = transforms.Compose([
    # 将图片转换为tensor
    transforms.ToTensor(),
    # 并归一化像素值，这里的值是MNIST数据集的均值和标准差
    torchvision.transforms.Normalize(
        (0.1307,), (0.3081,))
])
print("transform loaded！")
# 加载mnist数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
print("dataset loaded！")
# 取10%的样本作为训练集和测试集 因为前面已经取了train为true和false，所以是隔离的，不会重复，直接取10%就可以,用效率高的方法随机取10%
sub_train_dataset = torch.utils.data.Subset(train_dataset,
                                            random.sample(range(len(train_dataset)), int(len(train_dataset) * 0.1)))
sub_test_dataset = torch.utils.data.Subset(test_dataset,
                                           random.sample(range(len(test_dataset)), int(len(test_dataset) * 0.1)))
print("dataset created！")
# 转换为自定义数据集
sub_train_dataset = PairsDataset(sub_train_dataset)
sub_test_dataset = PairsDataset(sub_test_dataset)
print("dataset transformed！")
# 创建数据加载器
train_loader = DataLoader(sub_train_dataset, batch_size=batch_size, shuffle=True)
# 测试集的batch_size要大一点，因为测试集的数据量比较小，如果batch_size太小，会导致测试集的准确率不稳定
test_loader = DataLoader(sub_test_dataset, batch_size=2000, shuffle=True)
print("all data loaded！")


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 第一层卷积层
        self.conv1 = nn.Sequential(
            # 输入的有两张图片，所以输入通道为2
            # mat1 and mat2 shapes cannot be multiplied (64x2048 and 1568x100)
            # 这是因为卷积核的大小和步长的原因，导致卷积后的图片大小不是28*28，而是26*26，所以要改变padding的大小,应该改为1，因为只吞掉了中间的左边一格，如果kernel为5，padding应该为2

            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1),
            # bn层
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 池化层使用均值池化，池化核的大小为2*2，步长为2，这样池化后的图片大小变为原来的一半
            nn.AvgPool2d(kernel_size=2)
        )
        # 第二层卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        # 第三层卷积层
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 使用均值池化
            # nn.AvgPool2d(kernel_size=7)
        )
        # 全连接层
        # 这里的全连接层的输入是32*7*7，因为经过两次池化，图片的大小变为原来的1/4，所以是28/4=7
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 7 * 7, 100),
            # 用sigmoid函数来将输出限制在0-1之间
            nn.Sigmoid()
        )
        # 最后的输出层,sigmoid函数将输出限制在0-1之间
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        # 两张图片合并在一起
        # cat函数是将两个张量（tensor）拼接在一起，dim=1表示在第一个维度上拼接，即在通道上拼接
        x = torch.cat((x[0], x[1]), dim=1)
        # 卷积层
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 将卷积层的输出展平
        x = x.view(x.size(0), -1)
        # 全连接层
        x = self.fc1(x)
        # 输出层
        x = self.fc2(x)
        # sigmoid函数将输出限制在0-1之间
        # x = torch.softmax(x, dim=1)
        return x


# 初始化模型和优化器、损失函数
model = CNN().to(device)
print("model loaded！")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("optimizer loaded！")
# 二分类问题，所以用二分类的损失函数,这里用了sigmoid函数，所以用BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()
print("criterion loaded！")
# 训练模型
model.train()
for epoch in range(epochs):
    for i, (img1, img2, label) in enumerate(train_loader):
        img1 = img1.to(device)
        img2 = img2.to(device)

        # 前向传播

        output = model((img1, img2))
        # 将label的类型转换为Float，因为后面计算损失的时候需要
        label = label.type(torch.FloatTensor)
        # 将label的shape转换为和output一样的shape
        label = label.view(output.shape)
        label = label.to(device)
        # 计算损失
        loss = criterion(output, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 打印损失
        if i % 100 == 0:
            print('epoch: {}, step: {}, loss: {}'.format(epoch, i, loss.item()))
# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for i, (img1, img2, label) in enumerate(test_loader):
        img1 = img1.to(device)
        img2 = img2.to(device)

        # 前向传播
        output = model((img1, img2))
        # 计算准确率
        pred = torch.round(torch.sigmoid(output))
        # 将pred的shape转换为和label一样的shape
        pred = pred.view(label.shape)
        # 转为int类型
        pred = pred.type(torch.IntTensor)
        # 记得别丢gpu上了

        # debug 输出预测值和真实值
        # print("label:")
        # print(label)
        # print("pred:")
        # print(pred)
        correct += (pred == label).sum().item()
        total += label.size(0)
print('accuracy: {}'.format(correct / total))
