'''
Authored by Dr. Antonio Giuliano Zippo (Consiglio Nazionale delle Ricerche, antonio.zippo@gmail.com)
and
Dr. Alessandra Patera (University of Turin, alessandra.patera@to.infn.it)
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi

'''
Path to the volume directories
'''
basepath = '/media/antonio/1210-92BA/disk19/'
#volumes = ['388','389','390','404','405','406','396','397','398']
volumes = ['390']

for vol in volumes:
	print vol
	for i in range (1,10):
		print 'slice ',basepath + 'brain1__B' + vol + '/rec_8bit_pad/brain1__B' + vol +'000' + str(i) +'.rec.8bit.tif'
		img = cv2.imread(basepath + 'brain1__B' + vol + '/rec_8bit_pad/brain1__B' + vol +'000' + str(i) +'.rec.8bit.tif',0)

		destimg = np.zeros((img.shape))
		mask = np.zeros((img.shape))
		cv2.circle(mask,(1020,1020),1020,(255,255,255),-1,8,0)

		jac = np.diff(img)
		blurred = cv2.GaussianBlur(img, (5,5), 0)    
		lower = np.percentile(np.ndarray.flatten(jac),57)
		upper = 17*np.median(jac)   
		edges = cv2.Canny(blurred,lower,upper)   
		kernel = np.ones((15,15),np.uint8)
		dilation = cv2.dilate(edges,kernel)

		im_floodfill = dilation.copy()
		h, w = dilation.shape[:2]
		idk = np.where(dilation == 0)
		xx = idk[0]
		yy = idk[1]

		mask_flood = np.zeros((h+2, w+2), np.uint8)
		cv2.floodFill(im_floodfill, mask_flood, (xx[0],yy[0]), 255);
		im_floodfill_inv = cv2.bitwise_not(im_floodfill)
		im_out = dilation | im_floodfill_inv

		kernel = np.ones((4,4),np.uint8)
		erosion = cv2.erode(im_out,kernel,iterations = 5)

		idx = np.where(mask != 0)
		destimg[idx] = erosion[idx]

		dist = np.zeros((2040,2040))
		for r in range(2040):
			for j in range(2040):
				tmp = int(erosion[r,j]) - int(img[r,j])
				if tmp < 0:
					dist[r,j] = 0
				else:
					dist[r,j] = tmp

		C = plt.contour(dist)
		plt.close()
		levels = C.levels
	
		image_a = dist > 0
		image_b = dist <= levels[len(levels)-3]
		new_image = image_a.astype('uint8') & image_b.astype('uint8') 

		im_out = ndi.binary_fill_holes(new_image)
		iii = im_out.astype('uint8')
		idx = np.where(iii > 0)
		iii[idx[0],idx[1]] = 255

		final = ndi.filters.gaussian_filter(iii,1)		
				
		#np.save(basepath + 'brain1__B' + vol + '/rec_8bit_pad/aut_segmented/brain1_B' + vol + '000'+ str(i) +'.segm.png',final)
		cv2.imwrite(basepath + 'brain1__B' + vol + '/rec_8bit_pad/aut_segmented/brain1_B' + vol + '000'+ str(i) +'.segm.png', final)

	for i in range (10,100):
		print 'slice ',basepath + 'brain1__B' + vol + '/rec_8bit_pad/brain1__B' + vol +'00' + str(i) +'.rec.8bit.tif'
		img = cv2.imread(basepath + 'brain1__B' + vol + '/rec_8bit_pad/brain1__B' + vol +'00' + str(i) +'.rec.8bit.tif',0)

		destimg = np.zeros((img.shape))
		mask = np.zeros((img.shape))
		cv2.circle(mask,(1020,1020),1020,(255,255,255),-1,8,0)


		jac = np.diff(img)
		blurred = cv2.GaussianBlur(img, (5,5), 0)    
		lower = np.percentile(np.ndarray.flatten(jac),57)
		upper = 17*np.median(jac)   
		edges = cv2.Canny(blurred,lower,upper)   
		kernel = np.ones((15,15),np.uint8)
		dilation = cv2.dilate(edges,kernel)

		im_floodfill = dilation.copy()
		h, w = dilation.shape[:2]
		idk = np.where(dilation == 0)
		xx = idk[0]
		yy = idk[1]

		mask_flood = np.zeros((h+2, w+2), np.uint8)
		cv2.floodFill(im_floodfill, mask_flood, (xx[0],yy[0]), 255);
		im_floodfill_inv = cv2.bitwise_not(im_floodfill)
		im_out = dilation | im_floodfill_inv

		kernel = np.ones((4,4),np.uint8)
		erosion = cv2.erode(im_out,kernel,iterations = 5)

		idx = np.where(mask != 0)
		destimg[idx] = erosion[idx]

		dist = np.zeros((2040,2040))
		for r in range(2040):
			for j in range(2040):
				tmp = int(erosion[r,j]) - int(img[r,j])
				if tmp < 0:
					dist[r,j] = 0
				else:
					dist[r,j] = tmp		

		C = plt.contour(dist)
		plt.close()
		levels = C.levels
	
		image_a = dist > 0
		image_b = dist <= levels[len(levels)-3]
		new_image = image_a.astype('uint8') & image_b.astype('uint8') 

		im_out = ndi.binary_fill_holes(new_image)
		iii = im_out.astype('uint8')
		idx = np.where(iii>0)
		iii[idx[0],idx[1]] = 255

		final = ndi.filters.gaussian_filter(iii,1)

		#np.save(basepath + 'brain1__B' + vol + '/rec_8bit_pad/aut_segmented/brain1_B' + vol + '00'+ str(i) +'.segm.png',final)
		cv2.imwrite(basepath + 'brain1__B' + vol + '/rec_8bit_pad/aut_segmented/brain1_B' + vol + '00'+ str(i) +'.segm.png',final)

	for i in range (100,1000):
		print 'slice ',basepath + 'brain1__B' + vol + '/rec_8bit_pad/brain1__B' + vol +'0' + str(i) +'.rec.8bit.tif'		
		img = cv2.imread(basepath + 'brain1__B' + vol + '/rec_8bit_pad/brain1__B' + vol +'0' + str(i) +'.rec.8bit.tif',0)

		destimg = np.zeros((img.shape))
		mask = np.zeros((img.shape))
		cv2.circle(mask,(1020,1020),1020,(255,255,255),-1,8,0)


		jac = np.diff(img)
		blurred = cv2.GaussianBlur(img, (5,5), 0)    
		lower = np.percentile(np.ndarray.flatten(jac),57)
		upper = 17*np.median(jac)   
		edges = cv2.Canny(blurred,lower,upper)   
		kernel = np.ones((15,15),np.uint8)
		dilation = cv2.dilate(edges,kernel)

		im_floodfill = dilation.copy()
		h, w = dilation.shape[:2]
		idk = np.where(dilation == 0)
		xx = idk[0]
		yy = idk[1]

		mask_flood = np.zeros((h+2, w+2), np.uint8)
		cv2.floodFill(im_floodfill, mask_flood, (xx[0],yy[0]), 255);
		im_floodfill_inv = cv2.bitwise_not(im_floodfill)
		im_out = dilation | im_floodfill_inv

		kernel = np.ones((4,4),np.uint8)
		erosion = cv2.erode(im_out,kernel,iterations = 5)

		idx = np.where(mask != 0)
		destimg[idx] = erosion[idx]

		dist = np.zeros((2040,2040))
		for r in range(2040):
			for j in range(2040):
				tmp = int(erosion[r,j]) - int(img[r,j])
				if tmp < 0:
					dist[r,j] = 0
				else:
					dist[r,j] = tmp

		C = plt.contour(dist)
		plt.close()
		levels = C.levels
	
		image_a = dist > 0
		image_b = dist <= levels[len(levels)-3]
		new_image = image_a.astype('uint8') & image_b.astype('uint8') 

		im_out = ndi.binary_fill_holes(new_image)
		iii = im_out.astype('uint8')
		idx = np.where(iii>0)
		iii[idx[0],idx[1]] = 255

		final = ndi.filters.gaussian_filter(iii,1)

		#np.save(basepath + 'brain1__B' + vol + '/rec_8bit_pad/aut_segmented/brain1_B' + vol + '0' + str(i) +'.segm.png',final)
		cv2.imwrite(basepath + 'brain1__B' + vol + '/rec_8bit_pad/aut_segmented/brain1_B' + vol + '0' + str(i) +'.segm.png',final)

	for i in range (1000,2033):
		print 'slice ',basepath + 'brain1__B' + vol + '/rec_8bit_pad/brain1__B' + vol + str(i) +'.rec.8bit.tif'		
		img = cv2.imread(basepath + 'brain1__B' + vol + '/rec_8bit_pad/brain1__B' + vol + str(i) +'.rec.8bit.tif',0)

		destimg = np.zeros((img.shape))
		mask = np.zeros((img.shape))
		cv2.circle(mask,(1020,1020),1020,(255,255,255),-1,8,0)


		jac = np.diff(img)
		blurred = cv2.GaussianBlur(img, (5,5), 0)    
		lower = np.percentile(np.ndarray.flatten(jac),57)
		upper = 17*np.median(jac)   
		edges = cv2.Canny(blurred,lower,upper)   
		kernel = np.ones((15,15),np.uint8)
		dilation = cv2.dilate(edges,kernel)

		im_floodfill = dilation.copy()
		h, w = dilation.shape[:2]
		idk = np.where(dilation == 0)
		xx = idk[0]
		yy = idk[1]

		mask_flood = np.zeros((h+2, w+2), np.uint8)
		cv2.floodFill(im_floodfill, mask_flood, (xx[0],yy[0]), 255);
		im_floodfill_inv = cv2.bitwise_not(im_floodfill)
		im_out = dilation | im_floodfill_inv

		kernel = np.ones((4,4),np.uint8)
		erosion = cv2.erode(im_out,kernel,iterations = 5)

		idx = np.where(mask != 0)
		destimg[idx] = erosion[idx]

		dist = np.zeros((2040,2040))
		for r in range(2040):
			for j in range(2040):
				tmp = int(erosion[r,j]) - int(img[r,j])
				if tmp < 0:
					dist[r,j] = 0
				else:
					dist[r,j] = tmp

		C = plt.contour(dist)
		plt.close()
		levels = C.levels
	
		image_a = dist > 0
		image_b = dist <= levels[len(levels)-3]
		new_image = image_a.astype('uint8') & image_b.astype('uint8') 

		im_out = ndi.binary_fill_holes(new_image)
		iii = im_out.astype('uint8')
		idx = np.where(iii>0)
		iii[idx[0],idx[1]] = 255

		final = ndi.filters.gaussian_filter(iii,1)

		#np.save(basepath + 'brain1__B' + vol + '/rec_8bit_pad/aut_segmented/brain1_B' + vol + str(i) +'.segm.png',final)
		cv2.imwrite(basepath + 'brain1__B' + vol + '/rec_8bit_pad/aut_segmented/brain1_B' + vol + str(i) +'.segm.png',final)	






