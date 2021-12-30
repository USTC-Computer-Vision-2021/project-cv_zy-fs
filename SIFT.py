import cv2
import numpy as np


# 显示图片
def cv_show(name, result):
    cv2.imshow(name, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 利用SIFT算法提取特征
def featureExtra(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将彩色图像统一转换为灰度图像
    descriptor = cv2.xfeatures2d.SIFT_create()  # 使用了opencv库中的SIFT算法提取特征
    kps, features = descriptor.detectAndCompute(gray, None)
    #将返回的特征点转化为numpy格式的数组
    kps = np.float32([kp.pt for kp in kps])
    return kps, features


# 匹配特征点，取比率为0.8
def keyPointMatch(features1, features2):
    matcher = cv2.DescriptorMatcher_create('BruteForce')  # 设置蛮力匹配
    rawmatches = matcher.knnMatch(features1, features2, 2)
    goodmatches = []
    for m, n in rawmatches:
        if m.distance < 0.8 * n.distance:
            goodmatches.append(m)
    return goodmatches


# 连线画出两张图片的特征点匹配关系
def drawMatches(image1, image2, kps1, kps2, matches, status):
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    visResults = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
    visResults[0:h1, 0:w1] = image1
    visResults[0:h2, w1:] = image2
    matchIndex = []
    for m in matches:   # 将匹配对的index信息提取保存便于对比
        matchIndex.append((m.trainIdx, m.queryIdx))

    for ((trainIdx, queryIdx), s) in zip(matchIndex, status):
        # 当点对匹配成功时，画到结果图中
        if s == 1:
            # 画出匹配对
            pt1 = (int(kps1[queryIdx][0]), int(kps1[queryIdx][1]))
            pt2 = (int(kps2[trainIdx][0]) + w1, int(kps2[trainIdx][1]))
            cv2.line(visResults, pt1, pt2, (0, 255, 0), 1)
    cv_show('match', visResults)
    cv2.imwrite('./result/matched_img1.jpg', visResults)


# 图像拼接以及透明度融合
def stitchBlend(image1, image2):
    kps1, features1 = featureExtra(image1)
    kps2, features2 = featureExtra(image2)
    matches = keyPointMatch(features1, features2)
    if len(matches) > 4:    # 使用了RANCAC算法，需要至少四个特征点对
        pts1 = np.float32([kps1[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kps2[m.trainIdx] for m in matches]).reshape(-1, 1, 2)
        reprojThresh = 4    # RANSAC重投影阈值被设定为4
        # 使用RANSAC算法，并计算H变换矩阵
        H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC, reprojThresh)

        # 由于被融合进原图像的小图可能是和原图像任一小部分相匹配的，故利用变换矩阵H进行变换

        # 设计一个和被融合进原图像中的小图像相同大小的mask，用于融合和羽化边缘
        mask = np.ones((image1.shape[0], image1.shape[1], 3), dtype="float")
        mask_h = cv2.warpPerspective(mask, H, (image2.shape[1], image2.shape[0]))
        mask_lastcopy = mask_h.copy()
        for i in range(150):
            mask_blured = cv2.blur(mask_lastcopy, (40, 40))
            mask_lastcopy = mask_blured.copy()

        # 利用warpPerspective做透视变换以匹配合适的角度
        result1 = cv2.warpPerspective(image1, H, (image2.shape[1], image2.shape[0]))
        result1 = result1.astype(float)
        result1 = cv2.multiply(mask_blured, result1)  # 模糊蒙版用于羽化边缘
        cv_show('maskedImage1', result1/255)

        result2 = image2.astype(float)
        result2 = cv2.multiply(1-mask_blured, result2)  # 将原图像中待融合位置的权重降低，使得融合后更加突出待融合图片
        cv_show('maskedImage2', result2/255)
        resultImg = cv2.add(result1, result2)
        resultImg = resultImg/255
        cv_show('result', resultImg)
        resultImg = resultImg*255
        cv2.imwrite('./result/result1.jpg', resultImg)
        return resultImg, status


if __name__ == '__main__':
    image1 = cv2.imread('./source/test1old.jpg')  # 将待融合图片作为image1
    image2 = cv2.imread('./source/test1new.jpg')  # 将融合背景作为image2
    cv_show("image1", image1)
    cv_show("image2", image2)
    kps1, features1 = featureExtra(image1)
    kps2, features2 = featureExtra(image2)
    matches = keyPointMatch(features1, features2)
    result, status = stitchBlend(image1, image2)
    drawMatches(image1, image2, kps1, kps2, matches, status)