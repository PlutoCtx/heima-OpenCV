import cv2 as cv
def fac_detect_demo():
	# 将图片转为灰度图片
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	# 加载特征数据
	cv.CascadeClassifier('D:\Program Files\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')


# 加载图片
img = cv.imread("./image/face.jpg")
fac_detect_demo()
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()
