import numpy as np
import cv2
import os
import conf

""" returns batch, labels, and filelist """

class DATA():



	def __init__(self, dir_name):
		self.dir_path = os.path.join(conf.DATA_DIR, dir_name)
		self.filelist = os.listdir(self.dir_path)
		self.batch_size = conf.BATCH_SIZE
		self.size = len(self.filelist)
		self.useFullDataSize = true
		#self.dataSampleSize = #conf.SAMPLE_SIZE #to test data on smaller size if we'd like 
		self.data_index = 0


	# returns greyscale image and color image
	#
	#
	def read_img(self, filename):
		#loads img as is with alpha channels
		img = cv2.imread(filename, 3)
		height, width, channels = img.shape

		#resize image: input of the low-level features network to be of fixed size of 224 × 224
		img_resized = cv2.resize(img, (conf.IMAGE_SIZE, conf. IMAGE_SIZE))

		#convert img from RGB to LAB colorspace
		lab_img = cv2.cvtColor( img_resized, cv2.COLOR_BGR2Lab)

		grey_img = np.reshape(lab_img[:,:,0], (conf.IMAGE_SIZE, conf.IMAGE_SIZE,1))
		color_img = lab_img[:, :, 1:]


		return grey_img, color_img

	def get_batch(self):
		batch = []
		labels = []
		filelist = []

		for i in range(self.batch_size):
			filename = os.path.join(conf.DATA_DIR, self.dir_path, self.filelist[self.data_index])
			filelist.append(self.filelist[self.data_index])
			grey_img, color_img = self.read_img(filename)
			batch.append(grey_img)
			labels.append(color_img)
			self.data_index = (self.data_index + 1) % self.size #make sure it doesnt go over; consider using if statement instead

		#normalize values of images to be in range [0,1] of the sigmoid function
		batch = np.asarray(batch)/255
		labels = np.asarray(labels)/255


		return batch, labels, filelist













