# import cv2
# # 读一个图片进行显示
# lena = cv2.imread("imori.jpg")
# cv2.imshow("image", lena)
# cv2.waitKey(0)

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
# # 以灰度图的形式读取图像
# img = cv.imread("imori.jpg", 0)
# cv.imshow("image", img)
# # 在matplotlib中展示
# plt.imshow(img[:, :, ::-1])
# plt.title('匹配结果'), plt.xticks([]), plt.yticks([])
# plt.show()
# k = cv.waitKey(0)
# cv.imwrite("togrey.png",img)


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 1 读取图像
img = cv.imread("./imori.jpg")

# 2 显示图像
# 2.1 OPenCV
# cv.imshow("dili",img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# 2.2 matplotlib
plt.imshow(img, cmap=plt.cm.gray)
plt.show()

# 3 图像保存
cv.imwrite("../image/imori.png", img)

px = img[100, 100]
