"""
	检测人脸
"""
import cv2

# import cv2 as cv
# def face_detect_demo():
# 	# 将图片转为灰度图片
# 	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 	# 加载特征数据
# 	face_detector = cv.CascadeClassifier('D:\Program Files\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
# 	faces = face_detector.detectMultiScale(gray)
# 	for x, y, w, h in faces:
# 		cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
# 	cv.imshow('result', img)
# # 加载图片
# img = cv.imread("./image/wulin.jpeg")
# face_detect_demo()
# cv.waitKey(0)
# cv.destroyAllWindows()


"""
	检测多张人脸
"""
# import cv2 as cv
# # __all__ = [cv2]
# def face_detect_demo():
# 	# 将图片转为灰度图片
# 	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 	# 加载特征数据
# 	face_detector = cv.CascadeClassifier('D:\Program Files\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
# 	faces = face_detector.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=7, maxSize=(60, 60), minSize=(51, 51))
# 	for x, y, w, h in faces:
# 		print(x, y, w, h)
# 		cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
# 		cv.circle(img, center=(x+w//2, y+h//2), radius=w//2, color=(0, 255, 0), thickness=2)
# 	# 显示图片
# 	cv.imshow('result', img)
# # 加载图片
# img = cv.imread("./image/faces.png")
# face_detect_demo()
# cv.waitKey(0)
# cv.destroyAllWindows()


# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
#
# def face_detect_demo(img):
# 	# 图片灰度
# 	gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# 	# 加载特征数据
# 	face_detector = cv.CascadeClassifier('D:\Program Files\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_alt_tree.xml')
# 	faces = face_detector.detectMultiScale(gray)
#
#     # 修改检测参数 scaleFactor minNeighbors
#     # faces= face_detector.detectMultiScale(src, scaleFactor=1.01, minNeighbors=3)
#     for x,y,w,h in faces:
#         cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255), thickness=2)
#         cv.circle(src, center=(x+w//2, y+h//2), radius=w//2, color=(0, 255, 0), thickness=2)
#     # cv.imshow('result',src)
#
# # 读取视频
# cap = cv.VideoCapture('video.mp4')
# # 调用自己的摄像头
# cap = cv.VideoCapture(0)
# while True:
#     flag, frame = cap.read()
#     printf(flag, frame.shape)
#     face_detect_demo(frame)
#     cv.imshow('result', frame)
#     if ord('q') == cv.waitKey(10):
#         break
# cv.destroyAllWindows()
# cap.release()



# import cv2 as cv
# def face_detect_demo(img):
# 	# 图片灰度
# 	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 	# 加载特征数据
# 	face_detector = cv.CascadeClassifier("D:\Program Files\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_alt_tree.xml")
# 	faces = face_detector.detectMultiScale(gray)
# 	for x, y, w, h in faces:
# 		cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
# 		cv.circle(img, center=(x + w // 2, y + h // 2), radius=w // 2, color=(0, 255, 0), thickness=2)
# 	cv.imshow('result', img)
#
# # 读取视频
# cap = cv.VideoCapture('video.mp4')
# while True:
# 	flag, frame = cap.read()
# 	print('flag:', flag, 'frame.shape:', frame.shape)
# 	if not flag:
# 		break
# 	face_detect_demo(frame)
# 	if ord('q') == cv.waitKey(10):
# 		break
# cv.deatroyAllWindows()
# cap.release()
#


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def face_detect_demo(img):
	# 图片灰度
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	# 加载特征数据
	face_detector = cv.CascadeClassifier(
		"D:\Program Files\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_alt_tree.xml")
	faces = face_detector.detectMultiScale(gray)


	faces = face_detector.detectMultiScale(img)
	# faces = face_detector.detectMultiScale(img, scaleFactor=1.01, minNeighbors=3)

	for x, y, w, h in faces:
		cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
		cv.circle(img, center=(x + w // 2, y + h // 2), radius=w // 2, color=(0, 255, 0), thickness=2)
	cv.imshow('result', img)

# 读取视频
# cap = cv.VideoCapture('video.mp4')
# 调用自己的摄像头
cap = cv.VideoCapture(0)
while True:
	flag, frame = cap.read()
	print('flag:', flag, 'frame.shape:', frame.shape)
	if not flag:
		break
	face_detect_demo(frame)
	cv.imshow('result', frame)
	if ord('q') == cv.waitKey(10):
		break
cv.deatroyAllWindows()
cap.release()