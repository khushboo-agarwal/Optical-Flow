__author__ = 'khushboo_agarwal'

#import libraries
import math
from scipy import signal
from PIL import Image
import numpy as np
from numpy import *
from matplotlib import pyplot as plt
from pylab import *
import cv2
import random

'''
1. Lucas-Kanade method is basically an estimate of the movement of interesting
features in successive images of a scene. 

2. Lucas-Kanade assumes 
	a) 	that the intensity of the pixel does not change of a pixel
   		when it moves from frame1(I1) to frame(I2) with displacement (u,v):
   		I1(x,y) = I2(x+u, y+v) 
   	b) 	small motion, i.e (u,v)< 1 pixel
		expanding I2 in Taylor series, we get
		I2(x+u, y+v) = I2(x,y) + I2x(u) + I2y(v) + higher_order_terms
		             ~ I2(x,y) + I2x(u) + I2y(v)
    


3. Lucas-Kanade associate a movement vector (u,v) to 
every pixel in a scene obtained by comparing the two consecutive images or frames. 

4. works by trying to guess in which direction an object has moved so that local
changes in intensity can be explained. 

5. Optical Flow has a constraint equation to be solved;
	Ix.u + Iy.v +It = 0 ; (this equation is obtained by substituting 2b in 2a)
	where,
		Ix : derivative of I in x-direction
		Iy : derivative of I in y-direction
		It : derivative of I w.r.t time

		Ix and Iy are image gradients, and It is along t axis since now we deal with a 
		third dimension. These can be 
		d) This will only work for small movements

6. Smoothing the image first to attenuate any noise using Gaussian Smoothing as preprocessing

7. Using an averaging/smoothing filter of 2x2 size to find the first derivative of the smoothed 
	image. 

8. Finding features to track using goodFeatureToTrack
[Ref: http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html]

9. initialize the u and v vector array
'''

def LK_OpticalFlow(Image1, # Frame 1
Image2, # Frame 1
):
        '''
        This function implements the LK optical flow estimation algorithm with two frame data and without the pyramidal approach.
        '''
   	I1 = np.array(Image1)
	I2 = np.array(Image2)
	S = np.shape(I1)

	#applying Gaussian filter of size 3x3 to eliminate any noise
	I1_smooth = cv2.GaussianBlur(I1 #input image
								,(3,3)	#shape of the kernel
								,0      #lambda
								)
	I2_smooth = cv2.GaussianBlur(I2, (3,3), 0)

	'''
	let the filter in x-direction be Gx = 0.25*[[-1,1],[-1,1]]
	let the filter in y-direction be Gy = 0.25*[[-1,-1],[1,1]]
	let the filter in xy-direction be Gt = 0.25*[[1,1],[1, 1]]
	**1/4 = 0.25** for a 2x2 filter
	'''
		
	# First Derivative in X direction
	Ix = signal.convolve2d(I1_smooth,[[-0.25,0.25],[-0.25,0.25]],'same') + signal.convolve2d(I2_smooth,[[-0.25,0.25],[-0.25,0.25]],'same')
	# First Derivative in Y direction
	Iy = signal.convolve2d(I1_smooth,[[-0.25,-0.25],[0.25,0.25]],'same') + signal.convolve2d(I2_smooth,[[-0.25,-0.25],[0.25,0.25]],'same')
	# First Derivative in XY direction
	It = signal.convolve2d(I1_smooth,[[0.25,0.25],[0.25,0.25]],'same') + signal.convolve2d(I2_smooth,[[-0.25,-0.25],[-0.25,-0.25]],'same')
	
	# finding the good features
	features = cv2.goodFeaturesToTrack(I1_smooth # Input image
	,10000 # max corners
	,0.01 # lambda 1 (quality)
	,10 # lambda 2 (quality)
	)	

	feature = np.int0(features)

	plt.subplot(1,3,1)
	plt.title('Frame 1')
	plt.imshow(I1_smooth, cmap = cm.gray)
	plt.subplot(1,3,2)
	plt.title('Frame 2')
	plt.imshow(I2_smooth, cmap = cm.gray)#plotting the features in frame1 and plotting over the same
	for i in feature:
		x,y = i.ravel()
		cv2.circle(I1_smooth #input image
			,(x,y) 			 #centre
			,3 				 #radius
			,0 			 #color of the circle
			,-1 			 #thickness of the outline
			)
	
	#creating the u and v vector
	u = v = np.nan*np.ones(S)
	
	# Calculating the u and v arrays for the good features obtained n the previous step.
	for l in feature:
		j,i = l.ravel()
		# calculating the derivatives for the neighbouring pixels
		# since we are using  a 3*3 window, we have 9 elements for each derivative.
		
		IX = ([Ix[i-1,j-1],Ix[i,j-1],Ix[i-1,j-1],Ix[i-1,j],Ix[i,j],Ix[i+1,j],Ix[i-1,j+1],Ix[i,j+1],Ix[i+1,j-1]]) #The x-component of the gradient vector
		IY = ([Iy[i-1,j-1],Iy[i,j-1],Iy[i-1,j-1],Iy[i-1,j],Iy[i,j],Iy[i+1,j],Iy[i-1,j+1],Iy[i,j+1],Iy[i+1,j-1]]) #The Y-component of the gradient vector
		IT = ([It[i-1,j-1],It[i,j-1],It[i-1,j-1],It[i-1,j],It[i,j],It[i+1,j],It[i-1,j+1],It[i,j+1],It[i+1,j-1]]) #The XY-component of the gradient vector
		
		# Using the minimum least squares solution approach
		LK = (IX, IY)
		LK = np.matrix(LK)
		LK_T = np.array(np.matrix(LK)) # transpose of A
		LK = np.array(np.matrix.transpose(LK)) 
		
		A1 = np.dot(LK_T,LK) #Psedudo Inverse
		A2 = np.linalg.pinv(A1)
		A3 = np.dot(A2,LK_T)
		
		(u[i,j],v[i,j]) = np.dot(A3,IT) # we have the vectors with minimized square error
	
	#======= Pick Random color for vector plot========
	colors = "bgrcmykw"
	color_index = random.randrange(0,8)
	c=colors[color_index]
	#======= Plotting the vectors on the image========
	plt.subplot(1,3,3)
	plt.title('Vector plot of Optical Flow of good features')
	plt.imshow(I1,cmap = cm.gray)
	for i in range(S[0]):
		for j in range(S[1]):
			if abs(u[i,j])>t or abs(v[i,j])>t: # setting the threshold to plot the vectors
				plt.arrow(j,i,v[i,j],u[i,j],head_width = 5, head_length = 5, color = c)
				
	plt.show()

t = 0.3 # choose threshold value
#import the images
Image1 = Image.open('basketball1.png').convert('L')
Image2 = Image.open('basketball2.png').convert('L')
LK_OpticalFlow(Image1, Image2)
Image3 = Image.open('grove1.png').convert('L')
Image4 = Image.open('grove2.png').convert('L')
LK_OpticalFlow(Image3, Image4)
Image5 = Image.open('teddy1.png').convert('L')
Image6 = Image.open('teddy2.png').convert('L')
LK_OpticalFlow(Image5, Image6)

