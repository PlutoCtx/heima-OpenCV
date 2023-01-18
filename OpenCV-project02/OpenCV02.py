# @Version: python3.10
# @Time: 2023/1/18 9:42
# @Author: MaxBorooks
# @Email: chentingxian195467@163.com
# @File: OpenCV02.py
# @Software: PyCharm
# @User: chent

# 导入cv模块
import cv2 as cv
# 读取图片
img = cv.imread("../image/face.jpeg")

"""
	灰度转换
"""
# # 灰度转换
# gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# # 显示灰度图片
# cv.imshow("gray", gray_img)
# # 保存灰度图片
# cv.imwrite("gray_face.jpg", gray_img)
# # 显示图片
# cv.imshow("img_read", img)

"""
	修改图片
"""
# # 尺寸修改
# resize_img = cv.resize(img, dsize=(200, 200))
# # 显示图片
# cv.imshow("resize_img", resize_img)
# # 打印图片尺寸
# print("未修改：", img.shape)
# print("修改后：", resize_img.shape)

"""
	绘制矩形
"""
# x, y, w, h = 100, 100, 100, 100
# cv.rectangle(img, (x, y, x + w, y + h), color=(0, 0, 255), thickness=2)
# cv.circle(img, center=(x + w, y + h), radius=100, color=(255, 0, 0), thickness=2)
# cv.imshow('rec_img', img)


"""
	
"""
# 等待
while True:
	if ord('q') == cv.waitKey(0):
		break
# 释放资源
cv.destroyAllWindows()

