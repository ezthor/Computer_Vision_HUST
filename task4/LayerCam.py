import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
# 加载模型
model = torch.load("torch_alex.pth")
model.eval()


# 图像预处理
def preprocess_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


# 图像加载及预处理
image_path = './data4/both.jpg'
input_image = preprocess_image(image_path)


'''
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
# 选择目标层
target_layers = [model.features[9]]
# LayerCAM 可解释性分析

layer_cam = LayerCAM(model=model, target_layers=target_layers)
targets = [ClassifierOutputTarget(0)]  # 选择目标类别,有两个类别，则0,1可选
# 生成cam图
grayscale_cam = layer_cam(input_tensor=input_image,eigen_smooth=True)
# 保存热图
heatmap = grayscale_cam
# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
original_image = Image.open(image_path)
rgb_img = original_image
# 获得原始图的rgb转float32 并归一化后的版本
# Exception: The input image should np.float32 in the range [0, 1]
# 这是因为原始图像是 uint8 格式的，而 Grad-CAM 生成的热图是 float32 格式的，所以需要将原始图像转为 float32 格式。
rgb_img = np.float32(rgb_img) / 255
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
# 查看结果是否正确
'''
cv2.imshow('result', visualization)
cv2.waitKey(0)
'''

# 将热图转回uint8，方便展示
heatmap_np = np.uint8(255 * heatmap)  # 将热图转为 0-255 范围内的整数
# 将热图格式转为正常二维图片
# heatmap_np的格式是(1,224,224),而fromarray函数只接受二维数组，所以要squeeze来去除第一个维度,这里的参数axis是所去除维度的索引
heatmap_np_2ch = heatmap_np.squeeze(axis=0)

# 展示原始图像、热图和叠加图像
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original_image)
axes[0].set_title('Original Image')
axes[0].axis('off')
# 单通道图片的颜色映射 cmap是映射方式
axes[1].imshow(heatmap_np_2ch, cmap='jet')
axes[1].set_title('Heatmap')
axes[1].axis('off')

axes[2].imshow(visualization)
axes[2].set_title('Overlay')
axes[2].axis('off')

plt.tight_layout()
plt.show()


# 可视化原始图像
'''
plt.imshow(np.array(Image.open(image_path)))
plt.axis('off')
plt.show()
'''