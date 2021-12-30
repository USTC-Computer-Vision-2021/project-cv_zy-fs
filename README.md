[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=6631454&assignment_repo_type=AssignmentRepo)
# 基于图像拼接的A Look Into The Past
成员及分工：
- 周晔 PB18081670
  - 设计 coding 报告
- 傅石 PB18081631
  - 设计 coding 报告
 
 ## 问题描述：
 - 项目灵感：项目首先是源于国内外很多摄影作品的A Look Into The Past 将某一小部分融合进整个大的背景图片中实现历史与现实的交汇
 - 应用场景：这种技术已经广泛应用于大量现实场景比如全景照片拼接，摄影后期，图像增强等方面
 - 相关知识：具体到计算机视觉领域，这个问题主要涉及到数字图像的特征点提取与匹配，图像投影变换，图像融合等技术。

## 原理分析：
本次实验的实现主要是以下三个步骤：特征提取，特征匹配，图片融合：
### 1.特征提取：
特征提取采用了OpenCV库中的SIFT即尺度不变特征变换算法，提取了原图像中的尺度不变特征，这些特征对于旋转、尺度缩放、亮度变化均保持着良好的不变性，对视角变换、仿射变换、噪声也有一定程度的稳定性，并且区分性好、具有多量性、高速性和可拓展性。

提取SIFT特征主要经过以下几个步骤：

首先利用高斯金字塔对图像进行不断地降采样，同时也利用DOG金字塔检测图像中的关键点。

然后对于DOG金字塔中的关键点，采集关键点所在高斯金字塔的领域窗口内像素的梯度和方向分布特征，并用一组向量描述位置、尺度、方向三个特征，并且包含了周围一定范围内的像素，后面所有的图像数据操作都相对于关键点的方向、尺度、位置，保证了变换的尺度不变性。

上述提取出来的SIFT特征即可用于接下来的特征匹配。
### 2.特征匹配：
特征匹配采用关键点特征向量的欧式距离来作为两幅图像中关键点的相似性判定度量。取图1的某个关键点，通过遍历找到图像2中的距离最近的两个关键点。在这两个关键点中，如果最近距离除以次近距离小于某个阈值，则判定为一对匹配点。这是SIFT作者Lowe提出的算法。显然，这个阈值越低，匹配的特征点数目越少，但也会更加稳定，如果提取出的匹配点较多的情况下，可以适当减小阈值提高匹配精度。

此外，这里还采用了一种被称为RANSAC即随机抽样一致算法，提取出匹配效果最好的4个特征点对，RANSAC算法假设数据中包含正确数据和异常数据（或称为噪声）。正确数据记为内点（inliers），异常数据记为外点（outliers）。同时RANSAC也假设，给定一组正确的数据，存在可以计算出符合这些数据的模型参数的方法。该算法核心思想就是随机性和假设性，随机性是根据正确数据出现概率去随机选取抽样数据，根据大数定律，随机性模拟可以近似得到正确结果。假设性是假设选取出的抽样数据都是正确数据，然后用这些正确数据通过问题满足的模型，去计算其他点，然后对这次结果进行一个评分。它的流程是：
- 从样本集中随机抽选一个RANSAC样本，即4个匹配点对
- 根据这4个匹配点对计算变换矩阵M
- 根据样本集，变换矩阵M，和误差度量函数计算满足当前变换矩阵的一致集consensus，并返回一致集中元素个数
- 根据当前一致集中元素个数判断是否最优(最大)一致集，若是则更新当前最优一致集
- 更新当前错误概率p，若p大于允许的最小错误概率则重复(1)至(4)继续迭代，直到当前错误概率p小于最小错误概率

通过以上方法我们就用特征点构建了匹配关系，并用匹配关系得到了两个图片之间的变换矩阵H
### 3.图片融合
利用上面得到的图片变换矩阵，我们就已经可以将待融合图片定位到背景图片上了。但是如果单纯叠加或者是单纯替代背景图片相应位置的效果都很差，前者会导致相应位置出现重影乃至混乱的像素点，后者会导致边界出现生硬的接缝。

因此我们考虑设计一个待融合位置相同大小的掩膜，它具有不断模糊后的羽化边缘，叠加到待融合图片上就可以使得它的边缘具有渐淡效果。同时将原图片的相应位置叠加以相反的权值掩膜，这使得原图像融合位置的中心接近于无图像，而在边缘处具有原本特征。这样二者叠加之后就会使得融合较为自然。

```  Python
# 设计一个和被融合进原图像中的小图像相同大小的mask，用于融合和羽化边缘
        mask = np.ones((image1.shape[0], image1.shape[1], 3), dtype="float")
        mask_h = cv2.warpPerspective(mask, H, (image2.shape[1], image2.shape[0]))
        mask_lastcopy = mask_h.copy()
        for i in range(150):
            mask_blured = cv2.blur(mask_lastcopy, (40, 40))
            mask_lastcopy = mask_blured.copy()
```

<img src="https://github.com/USTC-Computer-Vision-2021/project-cv_zy-fs/blob/main/maskImg/mask1.png" width="400"><img src="https://github.com/USTC-Computer-Vision-2021/project-cv_zy-fs/blob/main/maskImg/mask2.png" width="400">

## 结果展示
这里展示了其中一个运行结果，左图是
