import cv2
import numpy as np


def to_hls(img):
	print('yo')
	return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)


hls_img = cv2.imread(r'C:\Users\vamsi\OneDrive\Pictures\tester7.jpg')
hls_img = to_hls(hls_img)
img_hls_yellow_bin = hls_img


#lower_yellow = np.array([15,115,30])
#upper_yellow = np.array([100,255,204]) 


lower_white = np.array([0,150,0])
upper_white = np.array([255,255,255])

#mask = cv2.inRange(hls_img, lower_yellow, upper_yellow)

mask2 = cv2.inRange(hls_img, lower_white, upper_white)

kernal = np.ones((15,15), np.float32)/225
smoothed = cv2.filter2D(mask2, -1, kernal)

median = cv2.medianBlur(mask2,25)

cannied = cv2.Canny(median, 200, 400)

newimg = smoothed



cv2.imshow('yo', median)
cv2.waitKey(0)
cv2.destroyAllWindows()


