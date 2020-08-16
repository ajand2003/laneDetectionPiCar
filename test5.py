import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img15 = cv2.imread(r'C:\Users\vamsi\OneDrive\Pictures\tester14.jpg')

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=1):
	line_image = np.zeros_like(frame)
	
	
	if lines is not None:
		for line in lines:
			for x1, y1, x2, y2 in line:
				print(x1)
				print(x2)
				cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
	line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
		
	
	return line_image


def detectLine(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, 
                                    np.array([]), minLineLength=8, maxLineGap=4)

    return line_segments


def to_hls(img):
	
	return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)


img = cv2.imread(r'C:\Users\vamsi\OneDrive\Pictures\tester7.jpg')
hls_img = to_hls(img)
img_hls_yellow_bin = hls_img

img10 = Image.open(r'C:\Users\vamsi\OneDrive\Pictures\tester7.jpg')
width, height = img10.size
#lower_yellow = np.array([15,115,30])
#upper_yellow = np.array([100,255,204]) 


lower_white = np.array([0,125,0])
upper_white = np.array([255,255,255])


lower_black = np.array([0, 0, 0])
upper_black = np.array([255, 100, 50])

mask = cv2.inRange(hls_img, lower_black, upper_black)

mask2 = cv2.inRange(hls_img, lower_white, upper_white)

kernal = np.ones((5,5), np.float32)/25
smoothed = cv2.filter2D(mask2, -1, kernal)

res = cv2.bitwise_and(img,img, mask = mask)


cannied = cv2.Canny(smoothed, 200, 400)

newimg = mask2 

cv2.circle(img, (950,800), 5, (0,255,255), -1)
cv2.circle(img, (1375,825), 5, (255,255,255), -1)
cv2.circle(img, (735,950), 5, (0,0,255), -1)
cv2.circle(img, (1530,1000), 5, (255,0,255), -1)


arr1 = np.float32([[930,800],[1385,825], [735,950], [1560,1000]])
arr2 = np.float32([[0,0], [500,0], [0,600], [500, 600]])

matrix = cv2.getPerspectiveTransform(arr1, arr2) 
result = cv2.warpPerspective(img, matrix, (500,600))


mask3 = cv2.inRange(result, lower_white, upper_white)
mask3 = cv2.medianBlur(mask3,25)

#mask3 = cv2.Canny(mask3, 200,400)

newimg2 = cv2.Canny(img, 200,400)

mask4 = cv2.inRange(hls_img, lower_white, upper_white)

cv2.imshow('yo', mask3)
cv2.waitKey(0)
cv2.destroyAllWindows()

#lines = detectLine(mask3)  
#newimg = display_lines(mask3, lines)
  
  
"""

for r,theta in lines[0]: 
      
   
    a = np.cos(theta) 
  
    b = np.sin(theta) 
      
  
    x0 = a*r 
      

    y0 = b*r 
      
    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta)) 
    x1 = int(x0 + 1000*(-b)) 
      
    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta)) 
    y1 = int(y0 + 1000*(a)) 
  
    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta)) 
    x2 = int(x0 - 1000*(-b)) 
      
    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta)) 
    y2 = int(y0 - 1000*(a)) 
      
    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2). 
    # (0,0,255) denotes the colour of the line to be  
    #drawn. In this case, it is red.  
    cv2.line(img,(x1,y1), (x2,y2), (0,0,255),2) 
      
# All the changes made in the input image are finally 
# written on a new image houghlines.jpg 


"""


"""
cv2.imshow('yo', newimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
