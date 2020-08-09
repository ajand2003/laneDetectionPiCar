import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



def to_hls(img):
	print('yo')
	return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def compute_hls_white_yellow_binary(rgb_img):
 
    hls_img = to_hls(rgb_img)
    
    
    img_hls_yellow_bin = np.zeros_like(hls_img[:,:,0])
    img_hls_yellow_bin[((hls_img[:,:,0] >= 15) & (hls_img[:,:,0] <= 35))
                 & ((hls_img[:,:,1] >= 30) & (hls_img[:,:,1] <= 204))
                 & ((hls_img[:,:,2] >= 115) & (hls_img[:,:,2] <= 255))                
                ] = 1
    
   
    img_hls_white_bin = np.zeros_like(hls_img[:,:,0])
    img_hls_white_bin[((hls_img[:,:,0] >= 0) & (hls_img[:,:,0] <= 255))
                 & ((hls_img[:,:,1] >= 200) & (hls_img[:,:,1] <= 255))
                 & ((hls_img[:,:,2] >= 0) & (hls_img[:,:,2] <= 255))                
                ] = 1
    
    
    img_hls_white_yellow_bin = np.zeros_like(hls_img[:,:,0])
    img_hls_white_yellow_bin[(img_hls_yellow_bin == 1) | (img_hls_white_bin == 1)] = 1

    return img_hls_white_yellow_bin




img = cv2.imread(r'C:\Users\vamsi\OneDrive\Pictures\tester9.png')
img2 = compute_hls_white_yellow_binary(img)
img3 = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

x = [580.0, 740.0, 1100.0, 270.0, 580.0]
y = [460.0, 460.0, 670.0, 670.0, 460.0]


img4 = cv2.Canny(img3, 200,400)
plt.imshow(img4)
plt.show()

R = img3[:,:,0]
"""
cv2.imshow('yo',R)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
