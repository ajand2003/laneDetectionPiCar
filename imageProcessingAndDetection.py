import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#from imageReciever import carControl


#video file can be obtained from this file in the raspberry pi system
#process each frame in the video recorded and find steering angle based on processed image



def startProcessing():
	cap = self.videoRecorder('../data/tmp/car_videodatestr.avi')
	while cap.isOpened():
	    ret,frame = cap.read()
	    cv2.imshow('window-name', frame)
	    processedFrame = processedImage(frame)
	    count = count + 1
	    if cv2.waitKey(10) & 0xFF == ord('q'):
	        break






#img15 = cv2.imread(r'C:\Users\vamsi\OneDrive\Pictures\tester7.jpg')
#img = processedImage(img15)




def videoRecorder(self, path):
	return cv2.VideoWriter(path, self.fourcc, 20.0, (320,240))


def showImageTest(img):
	cv2.imshow('yo', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()





def slidingWindow(hist, img, nwindows = 10, margin = 100, pixThreshold = 1):
		plt.show(hist)
		output = img
		
		midpoint = np.int(hist.shape[0]//2)
		leftPeak = np.argmax(hist[:midpoint])
		rightPeak = np.argmax(hist[midpoint:])
		
		
		windowHeight = np.int(img.shape[0]//9)
		
		nonzero = img.nonzero()
		lanePixelsY = np.array(nonzero[0])
		lanePixelsX = np.array(nonzero[1])
		
		leftLaneInd = []
		rightLaneInd = []
		
		leftBase = leftPeak
		rightBase = rightPeak
		
		for window in range(nwindows):
			winBot = img.shape[0] - (window+1) * windowHeight
			winHigh = img.shape[0] - window * windowHeight
			
			winLeftLeft = leftBase - margin
			winLeftRight = leftBase + margin
			
			winRightLeft = rightBase - margin
			winRightRight = rightBase + margin 
			
			
			validLeftPix = ((lanePixelsY >= winBot) & (lanePixelsY < winHigh) & 
			(lanePixelsX >= winLeftLeft) & (lanePixelsX < winLeftRight)).nonzero()[0]
			
			validRightPix = ((lanePixelsY >= winBot) & (lanePixelsY < winHigh) & 
			(lanePixelsX >= winRightLeft) & (lanePixelsX < winRightRight)).nonzero()[0]
			
			leftLaneInd.append(validLeftPix)
			rightLaneInd.append(validRightPix)
			
			startPointLeft = (winLeftLeft, winBot)
			endPointLeft = (winLeftRight, winHigh)
			
			
			startPointRight = (winRightLeft, winBot)
			endPointRight = (winRightRight, winHigh)
			
			color = (250, 0, 0)
			
			"""
			cv2.imshow('yo', output)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			"""
			
			
			cv2.rectangle(output, startPointLeft, endPointLeft, color, 3)
			cv2.rectangle(output, startPointRight, endPointRight, color, 3)
			
			
			
			
			if len(validLeftPix) > pixThreshold:
				leftBase = np.int(np.mean(lanePixelsX[validLeftPix]))
				
			if len(validRightPix) > pixThreshold:
				rightBase = np.int(np.mean(lanePixelsX[validRightPix]))
				
		leftLaneInd = np.concatenate(leftLaneInd)
		rightLaneInd = np.concatenate(rightLaneInd)
		
		
		leftx = lanePixelsX[leftLaneInd]
		lefty = lanePixelsY[leftLaneInd]
		
		rightx = lanePixelsX[rightLaneInd]
		righty = lanePixelsY[rightLaneInd]
		
		

		
		"""	
		cv2.imshow('yo', output)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		"""


		fitLeft = np.polyfit(lefty, leftx,2)
		fitRight = np.polyfit(righty, rightx,2)
		
		leftA = []
		leftA.append(fitLeft[0])
		leftB = []
		leftB.append(fitLeft[1])
		leftC = []
		leftC.append(fitLeft[2])
		
		
		
		fitLeft[0] = np.mean(leftA[-10:])
		fitLeft[1] = np.mean(leftB[-10:])
		fitLeft[2] = np.mean(leftC[-10:])
		
		
		
		
			
		rightA = []
		rightA.append(fitRight[0])
		rightB = []
		rightB.append(fitRight[1])
		rightC = []
		rightC.append(fitRight[2])
		
		
		
		fitRight[0] = np.mean(rightA[-10:])
		fitRight[1] = np.mean(rightB[-10:])
		fitRight[2] = np.mean(rightC[-10:])
		
		print(fitRight)
		
		ploty = np.linspace(0,img.shape[0]-1, img.shape[0])
		
		leftCurve = np.array([])
		rightCurve = np.array([])
		
		leftCurve = fitLeft[0] * ploty**2 + fitLeft[1] * ploty + fitLeft[2]
		rightCurve = fitRight[0] * ploty**2 + fitRight[1] * ploty + fitRight[2]

		output[lanePixelsY[lefty],lanePixelsX[leftx]] = [255,255,255]
		output[lanePixelsY[righty], lanePixelsX[rightx]] = [255, 255, 255]
		
		
		"""
		cv2.imshow('yo', output)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		"""
		
		return output, (fitLeft, fitRight), (leftCurve, rightCurve), ploty
		
def display_line_of_bestfit_curve(img, leftCurve, rightCurve, ploty):
	colorImg = np.zeros_like(img, dtype='uint8')
	leftSet = np.array([np.transpose(np.vstack([leftCurve, ploty]))])
	rightSet = np.array([np.flipud(np.transpose(np.vstack([rightCurve, ploty])))])
	points = np.hstack((leftSet, rightSet))
	
	cv2.fillPoly(colorImg, np.int_(points), (255,100,100))
	
	
	#cv2.polylines(img, np.int32(leftPoints),  True, (0,200,255),3)
	
	return colorImg
	


def display_lines(frame, lines, line_color=(255, 255, 255), line_width=20):
	line_image = np.zeros_like(frame)
	img32 = np.zeros((500,600,3))
	if lines is not None:
		for line in lines:
			for x1, y1, x2, y2 in line:
			
				cv2.line(img32, (x1, y1), (x2, y2), line_color, line_width)
	#line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
		
	
	return img32


def detectLine(cropped_edges):
   
    rho = 1  
    angle = np.pi / 180  
    min_threshold = 10  
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, 
                                    np.array([]), minLineLength=4, maxLineGap=2)

    return line_segments


def to_hls(img):
	
	return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	
	

def histogram(img):
	histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
	fig, ax = plt.subplots(1, 2, figsize=(15,4))
	ax[0].imshow(img, cmap='gray')
	ax[0].axis("off")
	ax[0].set_title("Binary Thresholded Perspective Transform Image")
	ax[1].plot(histogram)
	ax[1].set_title("Histogram Of Pixel Intensities (Image Bottom Half)")


	
	return histogram


def SteeringAngleImage(leftCurve, rightCurve, img, ploty):
	y_eval = np.max(ploty)
	laneCenterPosition = (leftCurve + rightCurve)/2
	midSet = np.array([np.transpose(np.vstack([laneCenterPosition, ploty]))])

	midSet2 = midSet

	pts = np.hstack((midSet2, midSet))
	left_curverad = ((1 + (2*leftCurve[0]*y_eval + leftCurve[1])**2)**1.5) / np.absolute(2*leftCurve[0])
	right_curverad = ((1 + (2*rightCurve[0]*y_eval + rightCurve[1])**2)**1.5) / np.absolute(2*rightCurve[0])
	
	cv2.fillPoly(img, np.int_(midSet), (0,0,0))

	"""
	startPoint =np.int_(img[0]/2), 600
	endPoint = np.int_(img[0]/2), 0
	"""

	cv2.line(img, (np.int_(img.shape[0]/2 + 50),0), (np.int_(img.shape[0]/2),600), (250,255,255), 1)

	print(img.shape[0]/2)

	cv2.imshow('yo', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return img
	
	



def processedImage(img):
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


	arr1 = np.float32([[900,800],[1385,825], [705,950], [1560,1000]])
	arr2 = np.float32([[0,0], [520,0], [0,600], [520, 600]])

	matrix = cv2.getPerspectiveTransform(arr1, arr2) 
	result = cv2.warpPerspective(img, matrix, (520,600))


	mask3 = cv2.inRange(result, lower_white, upper_white)
	mask3 = cv2.medianBlur(mask3,25)



	newimg2 = cv2.Canny(img, 200,400)
	mask4 = cv2.inRange(hls_img, lower_white, upper_white)
	edgedImg = cv2.Canny(mask3, 520, 600)
	thickEdge = display_lines(edgedImg, detectLine(edgedImg))


	hist = histogram(thickEdge)





	ouputImg, equations, curves, ploty = slidingWindow(hist, thickEdge)
	curveWithBestFits = display_line_of_bestfit_curve(thickEdge, curves[0], curves[1], ploty)
	droz = SteeringAngleImage(curves[0], curves[1], curveWithBestFits, ploty)


	return droz

	
	
	
	
def inversePerspectiveWarp(img): 
	arr1 = np.float32([[900,800],[1385,825], [705,950], [1560,1000]])
	arr2 = np.float32([[0,0], [520,0], [0,600], [520, 600]])
	imgSize = np.float32([(img.shape[1],img.shape[0])])
	
	fat = cv2.getPerspectiveTransform(arr1, arr2)
	warped = cv2.warpPerspective(img, fat, (520,600))
	cv2.imshow('yo', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return warped








cv2.imshow('yo', curveWithBestFits)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('yo', thickEdge)	
cv2.waitKey(0)
cv2.destroyAllWindows()

def testHoughLines(img):
	for r,theta in lines[0]: 
	      
	   
	    a = np.cos(theta) 
	    b = np.sin(theta) 
	      
	    x0 = a*r   
	    y0 = b*r 
	     
	    x1 = int(x0 + 1000*(-b))   
	    y1 = int(y0 + 1000*(a)) 
	    x2 = int(x0 - 1000*(-b))    
	    y2 = int(y0 - 1000*(a)) 
	      
	    cv2.line(img,(x1,y1), (x2,y2), (0,0,255),2) 
      
 




def testDisplay(newimg):
	cv2.imshow('yo', newimg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()





"""
#def perspectiveWarp(img):	
	


cv2.imshow('yo', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


#test15 = inversePerspectiveWarp(curveWithBestFits)


#resultBit = cv2.bitwise_and(img, img, mask3 = mask3)
#gray = cv2.cvtColor(resultBit, cv2.COLOR_BGR2GRAY) 
"""

