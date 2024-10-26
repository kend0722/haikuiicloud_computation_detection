import cv2
import numpy as np


def is_in_poly(p, polys):
    """
    射线法判断点是否在多边形内，
    :param p: 点坐标[x, y]
    :param polys: 多边形坐标[[x1, y1], [x2, y2], ..., [xn, yn]]
    :return: is_in bool类型，点是否在多边形内/上
    """

    if len(polys) == 0:
        return True
    px, py = p
    is_in = False
    for poly in polys:
        for i, corner in enumerate(poly):
            next_i = i + 1 if i + 1 < len(poly) else 0
            # 多边形一条边上的两个顶点
            x1, y1 = corner
            x2, y2 = poly[next_i]
            # 在顶点位置
            if (x1 == px and y1 == py) or (x2 == px and y2 == py):
                is_in = True
                return is_in
            if min(y1, y2) < py <= max(y1, y2):
                x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
                # 在边上
                if x == px:
                    is_in = True
                    return is_in
                # 射线与边相交
                elif x > px:
                    is_in = not is_in
    return is_in

def is_backlit(image):
    # 读取图像

    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算图像的直方图
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    # 判断直方图中亮度较高的部分是否超过阈值
    threshold = 0.8 * hist.max()
    # return '背光程度', hist[200:256].sum()
    mean_val = np.mean(hist)

    # 转换为yuv亮度通道
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    yuv_ = np.mean(y)

    # 计算图像对比度
    con = np.std(gray_image)
    return yuv_, con


def is_blurred(image):
    # 读取图像
    # image = cv2.imread(image_path)
    # 计算图像的拉普拉斯方差
    lap_var = cv2.Laplacian(image, cv2.CV_64F).var()
    # 根据拉普拉斯方差判断是否保留图像
    return lap_var

def calculate_insert_over_union(rect1, rect2):
    # 计算交集区域
    x_overlap = max(0, min(rect1[1][0], rect2[1][0]) - max(rect1[0][0], rect2[0][0]))
    y_overlap = max(0, min(rect1[1][1], rect2[1][1]) - max(rect1[0][1], rect2[0][1]))

    inter_area = x_overlap * y_overlap

    # 计算并集区域
    area1 = (rect1[1][0] - rect1[0][0]) * (rect1[1][1] - rect1[0][1])
    area2 = (rect2[1][0] - rect2[0][0]) * (rect2[1][1] - rect2[0][1])
    union = area1 + area2 - inter_area
    iou = inter_area / union
    # print('交并比',iou)
    return iou, inter_area / area1, inter_area / area2
