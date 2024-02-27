import torch
import torchvision.transforms as transforms
from PIL import Image, ImageChops
import matplotlib.pyplot as plt


# 加载预训练的AlexNet模型
model = torch.load("torch_alex.pth")
'''
模型结构图
AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=4096, out_features=2, bias=True)
  )
)
'''
# 选择最后一层卷积层的索引
conv_layer_index = 10  # 这是第11层，因为索引从0开始

# 用于保存每个通道的特征图,用activation这个字典来保存
activation = {}


def hook_fn(module, input, output):
    activation['value'] = output


# 注册forward hook
hook = model.features[conv_layer_index].register_forward_hook(hook_fn)

# 读取并预处理输入图像
image_path = './data4/dog.jpg'
image = Image.open(image_path)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

# 将模型设置为评估模式
model.eval()

# 前向传播
with torch.no_grad():
    model(input_batch)

# 移除注册的hook
hook.remove()

mode = 'mix'
if mode == 'mix':
    # 获取特征图
    feature_map = activation['value'][0]

    # 将256个通道的特征图叠加在一起,
    merged_feature = feature_map.sum(dim=0)

    # 显示融合后的特征图
    plt.imshow(merged_feature.cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.show()
elif mode == 'all':
    # 获取特征图
    feature_map = activation['value'][0]

    # 可视化每个通道的特征图，使用16x16个子图
    fig, axs = plt.subplots(16, 16, figsize=(16, 16))

    for i in range(feature_map.shape[0]):
        row = i // 16
        col = i % 16
        axs[row, col].imshow(feature_map[i].cpu().numpy(), cmap='viridis')
        axs[row, col].axis('off')

    plt.show()
elif mode == 'blend':
    # 获取特征图
    feature_map = activation['value'][0]

    # 使用PIL库的blend方法融合每个通道的特征图
    blended_feature = Image.fromarray(feature_map[0].cpu().numpy().astype('uint8'))

    for i in range(1, feature_map.shape[0]):
        blended_feature = ImageChops.blend(blended_feature,
                                           Image.fromarray(feature_map[i].cpu().numpy().astype('uint8')), alpha=0.05)

    # 显示融合后的特征图
    Image.fromarray(feature_map[0].cpu().numpy().astype('uint8')).show()
