import cv2 as cv
from natsort import natsorted
import glob
import numpy as np
import os


Folder_name = r"C:\Users\Nova DC\Desktop\Wild boar"
Extension = ".jpg"

#Add Light
def add_light(image, filename, gamma=1.0):
	f = Folder_name+"/"+filename
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

	image=cv.LUT(image, table)
	if gamma>=1:
		cv.imwrite(f + "/light-"+str(gamma) + Extension, image)
	else:
		cv.imwrite(f + "/dark-" + str(gamma) + Extension, image)

# Saturation
def saturation_image(image,filename, saturation):
	f = Folder_name+"/"+filename
	image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

	v = image[:, :, 2]
	v = np.where(v <= 255 - saturation, v + saturation, 255)
	image[:, :, 2] = v

	image = cv.cvtColor(image, cv.COLOR_HSV2BGR)
	cv.imwrite(f + "/saturation-" + str(saturation) + Extension, image)

def flip_image(image,dir):
	f = Folder_name+"/"+filename
	image = cv.flip(image, dir)
	cv.imwrite(f + "/flip-" + str(dir)+Extension, image)

# Rotation
def rotate_image(image, filename, deg):
	f = Folder_name+"/"+filename
	rows, cols,c = image.shape
	M = cv.getRotationMatrix2D((cols/2,rows/2), deg, 1)
	image = cv.warpAffine(image, M, (cols, rows))
	cv.imwrite(f + "/Rotate-" + str(deg) + Extension, image)

# Blur
def gausian_blur(image, filename, blur):
	f = Folder_name+"/"+filename
	image = cv.GaussianBlur(image,(5,5),blur)
	cv.imwrite(f+"/GausianBLur-"+str(blur)+Extension, image)

def averageing_blur(image,shift):
	f = Folder_name+"/"+filename
	image=cv.blur(image,(shift,shift))
	cv.imwrite(f + "/AverageingBLur-" + str(shift) + Extension, image)

def median_blur(image,shift):
	f = Folder_name+"/"+filename
	image=cv.medianBlur(image,shift)
	cv.imwrite(f + "/MedianBLur-" + str(shift) + Extension, image)

# Contrast
def contrast_image(image, filename, contrast):
	f = Folder_name+"/"+filename
	image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
	image[:,:,2] = [[max(pixel - contrast, 0) if pixel < 190 else min(pixel + contrast, 255) for pixel in row] for row in image[:,:,2]]
	image= cv.cvtColor(image, cv.COLOR_HSV2BGR)
	cv.imwrite(f + "/Contrast-" + str(contrast) + Extension, image)

#grayscale
def grayscale_image(image):
	f = Folder_name+"/"+filename
	image= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	cv.imwrite(f + "/Grayscale-" + Extension, image)

	grayscale_image(image)

filenames = [img for img in glob.glob(r"C:\Users\Nova DC\Desktop\Wild boar\*.jpg")]
filenames =  natsorted(filenames)

for file in filenames:
	# Get the filename witn extension
	base=os.path.basename(file)
	# Filename without extension
	filename = os.path.splitext(base)[0]
	
	# Join parent directory and directory that you want to create
	path = os.path.join(Folder_name, filename) 
	os.mkdir(path)

	image = cv.imread(file)
	
	flip_image(image,0)#horizontal
	flip_image(image,1)#vertical
	#flip_image(image,-1)#both

	add_light(image, filename, 2.5)
	add_light(image,filename, 0.5)
	
	saturation_image(image, filename, 50)
	
	rotate_image(image, filename, 10)
	#rotate_image(image, filename, 30)
	rotate_image(image, filename, 60)	
	rotate_image(image, filename, 90)
	#rotate_image(image, filename, 120)
	#rotate_image(image, filename, 150)
	rotate_image(image, filename, 180)	
	#rotate_image(image, filename, 210)
	#rotate_image(image, filename, 240)
	rotate_image(image, filename, 270)
	#rotate_image(image, filename, 300)	
	rotate_image(image, filename, 330)

	gausian_blur(image,filename, 2)
	gausian_blur(image,filename, 6)

	averageing_blur(image,4)
	#averageing_blur(image,4)
	averageing_blur(image,6)

	median_blur(image,3)
	#median_blur(image,5)
	median_blur(image,7)

	contrast_image(image,filename, 20)

	grayscale_image(image)

