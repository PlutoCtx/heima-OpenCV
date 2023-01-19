# @Version: python3.10
# @Time: 2023/1/18 9:42
# @Author: MaxBrooks
# @Email: chentingxian195467@163.com
# @File: OpenCV02.py
# @Software: PyCharm
# @User: chent

"""
	源代码
"""
"""
	01 读取图片
"""
# #导入cv模块
# import cv2 as cv
# #读取图片
# img = cv.imread('face1.jpg')
# #显示图片
# cv.imshow('read_img',img)
# #等待
# cv.waitKey(0)
# #释放内存
# cv.destroyAllWindows()

"""
	02灰度转换
"""
# #导入cv模块
# import cv2 as cv
# #读取图片
# img = cv.imread('face1.jpg')
# #灰度转换
# gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# #显示灰度图片
# cv.imshow('gray',gray_img)
# #保存灰度图片
# cv.imwrite('gray_face1.jpg',gray_img)
# #显示图片
# cv.imshow('read_img',img)
# #等待
# cv.waitKey(0)
# #释放内存
# cv.destroyAllWindows()

"""
	03修改尺寸
"""
# #导入cv模块
# import cv2 as cv
# #读取图片
# img = cv.imread('face1.jpg')
# #修改尺寸
# resize_img = cv.resize(img,dsize=(200,200))
# #显示原图
# cv.imshow('img',img)
# #显示修改后的
# cv.imshow('resize_img',resize_img)
# #打印原图尺寸大小
# print('未修改：',img.shape)
# #打印修改后的大小
# print('修改后：',resize_img.shape)
# #等待
# while True:
#     if ord('q') == cv.waitKey(0):
#         break
# #释放内存
# cv.destroyAllWindows()

"""
	04绘制矩形
"""
# #导入cv模块
# import cv2 as cv
# #读取图片
# img = cv.imread('face1.jpg')
# #坐标
# x,y,w,h = 100,100,100,100
# #绘制矩形
# cv.rectangle(img,(x,y,x+w,y+h),color=(0,0,255),thickness=1)
# #绘制圆形
# cv.circle(img,center=(x+w,y+h),radius=100,color=(255,0,0),thickness=5)
# #显示
# cv.imshow('re_img',img)
# while True:
#     if ord('q') == cv.waitKey(0):
#         break
# #释放内存
# cv.destroyAllWindows()

"""
	05人脸检测
"""
# #导入cv模块
# import cv2 as cv
# #检测函数
# def face_detect_demo():
#     gary = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#     face_detect = cv.CascadeClassifier('D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
#     face = face_detect.detectMultiScale(gary,1.01,5,0,(100,100),(300,300))
#     for x,y,w,h in face:
#         cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
#     cv.imshow('result',img)
#
# #读取图像
# img = cv.imread('face1.jpg')
# #检测函数
# face_detect_demo()
# #等待
# while True:
#     if ord('q') == cv.waitKey(0):
#         break
# #释放内存
# cv.destroyAllWindows()

"""
	06检测多个人脸
"""
# #导入cv模块
# import cv2 as cv
# #检测函数
# def face_detect_demo():
#     gary = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#     face_detect = cv.CascadeClassifier('D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
#     face = face_detect.detectMultiScale(gary)
#     for x,y,w,h in face:
#         cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
#     cv.imshow('result',img)
#
# #读取图像
# img = cv.imread('face2.jpg')
# #检测函数
# face_detect_demo()
# #等待
# while True:
#     if ord('q') == cv.waitKey(0):
#         break
# #释放内存
# cv.destroyAllWindows()

"""
	07视频中的人脸检测	
"""
# #导入cv模块
# import cv2 as cv
# #检测函数
# def face_detect_demo(img):
#     gary = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#     face_detect = cv.CascadeClassifier('D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
#     face = face_detect.detectMultiScale(gary)
#     for x,y,w,h in face:
#         cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
#     cv.imshow('result',img)
#
# #读取摄像头
# cap = cv.VideoCapture(0)
# #循环
# while True:
#     flag,frame = cap.read()
#     if not flag:
#         break
#     face_detect_demo(frame)
#     if ord('q') == cv.waitKey(1):
#         break
# #释放内存
# cv.destroyAllWindows()
# #释放摄像头
# cap.release()





















# # 导入cv模块
# # import cv2 as cv
# # 读取图片
# # img = cv.imread("../image/face.jpeg")
#
# """
# 	灰度转换
# """
# # # 灰度转换
# # gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# # # 显示灰度图片
# # cv.imshow("gray", gray_img)
# # # 保存灰度图片
# # cv.imwrite("gray_face.jpg", gray_img)
# # # 显示图片
# # cv.imshow("img_read", img)
#
# """
# 	修改图片
# """
# # # 尺寸修改
# # resize_img = cv.resize(img, dsize=(200, 200))
# # # 显示图片
# # cv.imshow("resize_img", resize_img)
# # # 打印图片尺寸
# # print("未修改：", img.shape)
# # print("修改后：", resize_img.shape)
#
# """
# 	绘制矩形
# """
# # x, y, w, h = 100, 100, 100, 100
# # cv.rectangle(img, (x, y, x + w, y + h), color=(0, 0, 255), thickness=2)
# # cv.circle(img, center=(x + w, y + h), radius=100, color=(255, 0, 0), thickness=2)
# # cv.imshow('rec_img', img)
#
# """
# 	人脸录入
# """
# # # 摄像头
# # cap = cv.VideoCapture(0)
# #
# # flag = 1
# # num = 1
# #
# # while(cap.isOpened()):
# # 	ret_flag, Vshow = cap.read()
# # 	cv.imshow('Capture_Test', Vshow)
# # 	k = cv.waitKey(1) & 0xFF
# # 	if k == ord('s'):
# # 		cv.imwrite("../data/jm02/" + str(num) + ".name" + ".jpg", Vshow)
# # 		print("success to save " + str(num) + ".jpg")
# # 		print("**************************")
# # 		num += 1
# # 	elif k == ord(' '):
# # 		break
# #
# # cap.release()
# #
# #
# # # # 等待
# # # while True:
# # # 	if ord('q') == cv.waitKey(0):
# # # 		break
# # # 释放资源
# # cv.destroyAllWindows()
#
# """
# 	训练数据
# """
# # import os
# # import cv2
# # import numpy as np
# # import sys
# # from PIL import Image
# #
# # detector = cv2.CascadeClassifier("D:\Program Files\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml")
# #
# # def getImageAndLabels(path):
# # 	imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
# # 	faceSamples = []
# # 	ids = []
# # 	# 检测人脸
# # 	# 遍历列表中的图片
# # 	for imagePath in imagePaths:
# # 		# 打开图片
# # 		PIL_img = Image.open(imagePath).convert('L')	# convert it to grayscale
# # 		img_numpy = np.array(PIL_img, 'uint8')
# #
# # 		faces = detector.detectMultiScale(img_numpy)
# # 		# 获取每张图片的id
# # 		id = int(os.path.split(imagePath)[-1].split(".")[0])
# # 		# print(os.path.split(imagePath))
# # 		for (x, y, w, h) in faces:
# # 			faceSamples.append(img_numpy[y: y + h, x: x + w])
# # 			ids.append(id)
# #     # print("id:", id)
# #     # print("fs:", faceSamples)
# # 	return faceSamples, ids
# #
# # if __name__ == '__main__':
# # 	# 图片路径
# # 	path = './data/jm02/'
# # 	faces, ids = getImageAndLabels(path)
# # 	# 获取循环对象
# # 	recognizer = cv2.face.LBPHFaceRecognizer_create()
# # 	recognizer.train(faces, np.array(ids))
# # 	# Save the model into trainer/trainer.yml
# # 	recognizer.write('./data/trainer02.yml')
#
#
#
# import cv2 as cv
# import os
# import urllib
# import urllib.request
#
# # 加载训练数据集
# recognizer = cv.face.LBPHFaceRecognizer_create()
# # 加载数据
# recognizer.read('./data/trainer02.yml')
# # 名称
# names = []
# # 全局变量
# warningtime = 0
#
# # def md5(str):
#
#
#
#
#
# def face_detect_demo(img):
# 	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 	face_detector = cv.CascadeClassifier("D:\Program Files\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml")
# 	face = face_detector.detectMultiScale(gray, 1.1, 5, cv.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))
# 	for (x, y, w, h) in face:
# 		cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
# 		cv.circle(img, center=(x + w // 2, y + h // 2), radius=w // 2, color=(0, 255, 0), thickness=2)
# 		#人脸识别
# 		id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
# 		# print('标签id:', id, '置信评分:', confidence)
# 		if confidence > 80:
# 			global warningtime
# 			warningtime += 1
# 			if warningtime > 100:
# 				warning()
# 				warningtime = 0
# 			cv.putText(img, 'unknown', (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
# 		else:
# 			cv.putText(img, str(names[id]), (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
#
#
# # font = cv2.FONT_HERSHEY_SIMPLEX
# # id = 0
# # # 准备识别的图片
# # img = cv2.imread('./data/jm/1.pgm') #识别的图片
# #
# # faces = faceCascade.
# # for(x, y, w, h) in faces:
# #     cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
# #     #人脸识别
# #     id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
# #     print('标签id:', id, '置信评分:', confidence)
# # cv2.imshow('camera', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

"""
	08拍照保存
"""
# # 导入模块
# import cv2
# # 摄像头
# cap = cv2.VideoCapture(0)
#
# flag = 1
# num = 1
#
# while(cap.isOpened()):	# 检测是否在开启状态
#     ret_flag, Vshow = cap.read()	# 得到每帧图像
#     cv2.imshow("Capture_Test", Vshow)	# 显示图像
#     k = cv2.waitKey(1) & 0xFF	# 按键判断
#     if k == ord('s'):	# 保存
#        cv2.imwrite("../data/jm02/"+str(num)+".name"+".jpg", Vshow)
#        print("success to save"+str(num)+".jpg")
#        print("-------------------")
#        num += 1
#     elif k == ord(' '):		# 退出
#         break
# # 释放摄像头
# cap.release()
# # 释放内存
# cv2.destroyAllWindows()

"""
	09数据训练
"""
# import os
# import cv2
# import sys
# from PIL import Image
# import numpy as np
#
# def getImageAndLabels(path):
#     facesSamples = []
#     ids = []
#     imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
#     # 检测人脸
#     face_detector = cv2.CascadeClassifier('D:\Program Files\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml')
#     # 打印数组imagePaths
#     print('数据排列：', imagePaths)
#     # 遍历列表中的图片
#     for imagePath in imagePaths:
#         # 打开图片,黑白化
#         PIL_img = Image.open(imagePath).convert('L')
#         # 将图像转换为数组，以黑白深浅
#         # PIL_img = cv2.resize(PIL_img, dsize=(400, 400))
#         img_numpy=np.array(PIL_img, 'uint8')
#         # 获取图片人脸特征
#         faces = face_detector.detectMultiScale(img_numpy)
#         # 获取每张图片的id和姓名
#         id = int(os.path.split(imagePath)[1].split('.')[0])
#         # 预防无面容照片
#         for x,y,w,h in faces:
#             ids.append(id)
#             facesSamples.append(img_numpy[y:y+h,x:x+w])
#         # 打印脸部特征和id
#         # print('fs:', facesSamples)
#         print('id:', id)
#         # print('fs:', facesSamples[id])
#     print('fs:', facesSamples)
#     # print('脸部例子：',facesSamples[0])
#     # print('身份信息：',ids[0])
#     return facesSamples,ids
#
# if __name__ == '__main__':
#     # 图片路径
#     path = '../data/jm02/'
#     # 获取图像数组和id标签数组和姓名
#     faces, ids = getImageAndLabels(path)
#     # 获取训练对象
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     # recognizer.train(faces,names)#np.array(ids)
#     recognizer.train(faces, np.array(ids))
#     # 保存文件
#     recognizer.write('../data/jm02/trainer.yml')
#     # save_to_file('names.txt',names)

"""
	10人脸识别
"""
# import cv2
# import numpy as np
# import os
# # coding=utf-8
# import urllib
# import urllib.request
# import hashlib
#
# #加载训练数据集文件
# recogizer = cv2.face.LBPHFaceRecognizer_create()
# recogizer.read('../data/trainer02.yml')
# names = []
# warningtime = 0
#
# def md5(str):
#     import hashlib
#     m = hashlib.md5()
#     m.update(str.encode("utf8"))
#     return m.hexdigest()
#
# statusStr = {
#     '0': '短信发送成功',
#     '-1': '参数不全',
#     '-2': '服务器空间不支持,请确认支持curl或者fsocket,联系您的空间商解决或者更换空间',
#     '30': '密码错误',
#     '40': '账号不存在',
#     '41': '余额不足',
#     '42': '账户已过期',
#     '43': 'IP地址限制',
#     '50': '内容含有敏感词'
# }
#
#
# def warning():
#     smsapi = "http://api.smsbao.com/"
#     # 短信平台账号
#     user = 'pluto_chen'
#     # 短信平台密码
#     password = md5('Plutochen')
#     # 要发送的短信内容
#     content = '【报警】\n原因：检测到未知人员\n地点：xxx'
#     # 要发送短信的手机号码
#     phone = '19511310120'
#
#     data = urllib.parse.urlencode({'u': user, 'p': password, 'm': phone, 'c': content})
#     send_url = smsapi + 'sms?' + data
#     response = urllib.request.urlopen(send_url)
#     the_page = response.read().decode('utf-8')
#     print(statusStr[the_page])
#
# # 准备识别的图片
# def face_detect_demo(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 转换为灰度
#     face_detector = cv2.CascadeClassifier('D:\Program Files\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml')
#     face = face_detector.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))
#     # face=face_detector.detectMultiScale(gray)
#     for x, y, w, h in face:
#         cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=2)
#         cv2.circle(img, center=(x+w//2, y+h//2), radius=w//2, color=(0, 255, 0), thickness=1)
#         # 人脸识别
#         ids, confidence = recogizer.predict(gray[y:y + h, x:x + w])
#         # print('标签id:',ids,'置信评分：', confidence)
#         if confidence > 80:
#             global warningtime
#             warningtime += 1
#             if warningtime > 100:
#                warning()
#                warningtime = 0
#             cv2.putText(img, 'unknown', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
#         else:
#             cv2.putText(img, str(names[ids-1]), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
#     cv2.imshow('result', img)
#     #print('bug:',ids)
#
# def name():
#     path = '../data/jm02/'
#     #names = []
#     imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
#     for imagePath in imagePaths:
#        name = str(os.path.split(imagePath)[1].split('.', 2)[1])
#        names.append(name)
#
# cap = cv2.VideoCapture(0)
# # cap = cv2.VideoCapture('../data/1.mp4')
# name()
# while True:
#     flag, frame = cap.read()
#     if not flag:
#         break
#     face_detect_demo(frame)
#     if ord(' ') == cv2.waitKey(10):
#         break
# cv2.destroyAllWindows()
# cap.release()
# #print(names)

"""
	11 网页视频
"""
import cv2
class CaptureVideo(object):
	def net_video(self):
		# 获取网络视频流
		# cam = cv2.VideoCapture("rtmp://192.168.0.10/live/test")
		cam = cv2.VideoCapture("rtmp://58.200.131.2:1935/livetv/hunantv")
		while cam.isOpened():
			sucess, frame = cam.read()
			cv2.imshow("Network", frame)
			cv2.waitKey(1)
if __name__ == "__main__":
	capture_video = CaptureVideo()
	capture_video.net_video()


