import torch
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
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


# Grad-CAM 可解释性分析
# 目标网络的网络结构图
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
# 调用gradcam，参考官方给出的示例
# 选择目标层
target_layers = [model.features[-2]]

grad_cam = GradCAM(model=model, target_layers=target_layers)  # 选择目标层
# 这里的参数是选择你的列表里的第几个，这里有范围0和1
targets = [ClassifierOutputTarget(0)]  # 选择目标类别
# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
# targets如果是0那么是分数最高的
grayscale_cam = grad_cam(input_tensor=input_image)
heatmap=grayscale_cam

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
original_image = Image.open(image_path)
rgb_img = original_image
#rgb_img = np.array(original_image.convert("RGB"))
# Exception: The input image should np.float32 in the range [0, 1]
# 这是因为原始图像是 uint8 格式的，而 Grad-CAM 生成的热图是 float32 格式的，所以需要将原始图像转为 float32 格式。
rgb_img = np.float32(rgb_img) / 255
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
cv2.imshow('result', visualization)
cv2.waitKey(0)

# 生成热图

# numpy.AxisError: axis 2 is out of bounds for array of dimension 2
# 这是因为原始图像是单通道的，而 Grad-CAM 生成的热图是 3 通道的，所以需要将原始图像转为 3 通道的图像。
# 这里主要是层选错了
# 可视化热图
# plt.matshow(heatmap.squeeze())
# cv2.waitKey(0)

# 将热图叠加在原始图像上进行可视化展示
# AttributeError: 'numpy.ndarray' object has no attribute 'numpy'
# 这是因为 heatmap 是 numpy 格式的数组，而 matplotlib.pyplot.imshow() 函数只接受 PIL.Image.Image 或者 np.ndarray 格式的图像。
# heatmap_np = heatmap.squeeze().numpy()  # 将热图转为 numpy 格式
# heatmap_np = np.uint8(255 * heatmap)  # 将热图转为 0-255 范围内的整数

# TypeError: Invalid shape (1, 224, 224) for image data
# 这是因为 plt.imshow() 函数只接受 3 通道的图像，而 heatmap_np 是单通道的图像。所以应该将 heatmap_np 转为 3 通道的图像。
# 将热图覆盖到原始图像上
'''
Traceback (most recent call last):
  File "G:\计算机视觉课\实验\实验4\实验四模型和测试图片（PyTorch）\try.py", line 87, in <module>
    heatmap_img = Image.fromarray(heatmap_np, 'L').resize((224, 224))
  File "D:\anaconda\envs\Radar2023_coex_video\lib\site-packages\PIL\Image.py", line 3094, in fromarray
    raise ValueError(msg)
ValueError: Too many dimensions: 3 > 2.
'''
# 这个报错是因为 heatmap_np 是 3 维的数组，而 Image.fromarray() 函数只接受 2 维的数组。


# 读取原始图像

''''''
# 叠加热图在原始图像上
heatmap_np = np.uint8(255 * heatmap)  # 将热图转为 0-255 范围内的整数
# heatmap_np的格式是(1,224,224),而fromarray函数只接受二维数组，所以要squeeze来去除第一个维度,这里的参数axis是所去除维度的索引
heatmap_np_2ch = heatmap_np.squeeze(axis=0)
# 这里的L是指8位像素黑白图像，也就是灰度图像，全称是Luminance
heatmap_img = Image.fromarray(heatmap_np_2ch, 'L').resize(original_image.size)
# 备份一份灰度图
heatmap_img_gray=heatmap_img
# 将灰度图转为RGB图
heatmap_img = heatmap_img.convert('RGB')

# 调整热图透明度并叠加在原始图像上 alpha的值越大，热图的颜色越深，原始图像的颜色越浅,靠近0为原图，靠近1为热图
# 我打算使用灰度图高的地方透明，低的地方为黑色蒙住的方案,而不只是blend，应该使用mask，这样才能保证黑色的部分不透明
# 难搞，搞出来的太暗了，不好看

# 直接混合方法
'''
alpha = 0.8
blended = Image.blend(original_image, heatmap_img, alpha)
# blended.show()
'''
# 尝试mask方法
'''
# 这里的mask是一个二维数组，每个元素都是0-255的数，0表示完全透明，255表示完全不透明，所以这里的mask是和heatmap一样的
mask = np.array(heatmap_np_2ch)
print("mask:",mask)
print("mask_shape",mask.shape)
# IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed
# 这里不需要转换为3通道的图像，因为mask只是用来控制透明度的，不需要显示出来
# 这里调整阈值，大于阈值的为255，小于阈值的为0
threshold = 100
mask = np.where(mask > threshold, 255, 0)
mask = Image.fromarray(mask, 'L')
# debug调阈值 有点诡异
cv2.imshow("mask",mask)
cv2.waitkey(0)
exit()
# mask.show()
# 将mask和原图叠加
blended = Image.composite(original_image, heatmap_img, mask)

'''
'''
# 透明度方法
# 获取灰度图的数据 这里的gray_data是一个一维数组，每个元素都是0-255的数，0表示完全透明，255表示完全不透明
gray_data = heatmap_img.getdata()

# 根据灰度值创建透明度图 亮度越高，透明度越低
alpha_data = [255 - int(value) for value in gray_data]

# 创建透明度图
alpha_image = Image.new("L", original_image.size)
alpha_image.putdata(alpha_data)

# 将原始图像进行备份
ori_image = original_image
# 将透明度信息应用到原始图像
original_image.putalpha(alpha_image)

# 显示结果
original_image.show()
'''
# 展示原始图像、热图和叠加图像
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original_image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(heatmap_np_2ch, cmap='jet')
axes[1].set_title('Heatmap')
axes[1].axis('off')

axes[2].imshow(visualization)
axes[2].set_title('Overlay')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# Grad-CAM++ 可解释性分析

# 可视化原始图像
'''
plt.imshow(np.array(Image.open(image_path)))
plt.axis('off')
plt.show()
'''