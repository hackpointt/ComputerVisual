#导入关键的包
import numpy as np
import cv2

''' pts 为矩形的四点(x,y)pts列表
    
    初始一个坐标系list，它将会被组织成
    左上 右上 右下 左下 的形式
    [tl
     tr
     br
     bl]
'''
def order_points(pts):
    
    rect = np.zeros((4, 2),dtype = "float32")

    #左上点为最小和(smallest sum)，反之右上点为最大和
    s = pts.sum(axis = 1)           #列相加
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]


    diff = np.diff(pts, axis = 1)   #列相减
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    #将4个坐标排序
    rect = order_points(pts)
    #计算新图像的宽度
    #这个值为右下x坐标与左下x坐标之间的距离
    #或者右上与左上x坐标的距离，两者的最大值
    diffMat = rect - np.vstack((np.tile(rect[1:],(1,1)),rect[0]))
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5
    
    #得到变换后图像的宽度和高度
    maxWidth = int(max(distances[0::2]))
    maxHeight = int(max(distances[1::2]))

    #得到新图像的尺寸之后，构造出它的坐标点，还是以之前的顺序组织
    dst = np.array([
                   [0,0],
                    [maxWidth -1, 0],
                    [maxWidth-1,maxHeight -1],
                    [0,maxHeight -1]],dtype ="float32")

    #计算透视变换矩阵，并且实现它
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    #返回变换后的图像
    return warped




