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
import cv2 as cv
__all__ = [cv2]
def face_detect_demo():
	# 将图片转为灰度图片
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	# 加载特征数据
	face_detector = cv.CascadeClassifier('D:\Program Files\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
	faces = face_detector.detectMultiScale(gray)
	for x, y, w, h in faces:
		cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
		cv.circle(img, center=(x+w//2, y+h//2), radius=w//2, color=(0, 255, 0), thickness=2)
	# 显示图片
	cv.imshow('result', img)
# 加载图片
img = cv.imread("./image/faces.png")
face_detect_demo()
cv.waitKey(0)
cv.destroyAllWindows()
