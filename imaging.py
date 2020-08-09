from PIL import Image
import cv2
import numpy as np
import math

def length_of_line_segment(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # degree in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=10,
                                    maxLineGap=1)

    if line_segments is not None:
        for line_segment in line_segments:
            print('detected line_segment:')
            print("%s of length %s" % (line_segment, length_of_line_segment(line_segment[0])))

    return line_segments
	


image = cv2.imread(r'C:\Users\vamsi\OneDrive\Pictures\tester6.png')
test = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


lower_blue = np.array([120, 40, 40])
upper_blue = np.array([180, 255, 255])
mask = cv2.inRange(test, lower_blue, upper_blue)

 
edges = cv2.Canny(mask, 200,400)

cv2.imshow('yo' , edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

lines = detect_line_segments(edges)
if lines != None:
	print('yessir')




	
