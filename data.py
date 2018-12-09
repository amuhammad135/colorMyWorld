import numpy as np
import cv2
import os
import conf


class Data:

	def __init__(self, dir_name):
		self.dir_path = dir_name
		self.images = os.listdir(dir_name)
		self.batch_size = conf.BATCH_SIZE
		self.size = len(self.images)
		self.useFullDataSize = True
		# self.dataSampleSize = #conf.SAMPLE_SIZE #to test data on smaller size if we'd like
		self.data_index = 0

	# returns grey-scale image and color image
	def grey_color(self, image):
		# loads img as is with alpha channels
		img = cv2.imread(image, 3)

		# resize image: input of the low-level features network to be of fixed size of 224 × 224
		img_resized = cv2.resize(img, (conf.IMAGE_SIZE, conf. IMAGE_SIZE))

		# convert img from RGB to LAB color space
		lab_img = cv2.cvtColor( img_resized, cv2.COLOR_BGR2Lab)

		grey_img = np.reshape(lab_img[:, :, 0], (conf.IMAGE_SIZE, conf.IMAGE_SIZE, 1))
		color_img = lab_img[:, :, 1:]

		return grey_img, color_img

	def get_batch(self):
		batch = []
		labels = []
		images = []

		for i in range(self.batch_size):
			image = os.path.join(self.dir_path, self.images[self.data_index])
			images.append(self.images[self.data_index])
			grey_img, color_img = self.grey_color(image)
			batch.append(grey_img)
			labels.append(color_img)

			# Good catch! I think the mod is here so that it doesn't go over. We should probably use an if statement tho
			# I think it's more intuitive

			self.data_index = (self.data_index + 1) % len(self.images)

		# normalize values of images to be in range [0,1] of the sigmoid function
		batch = np.asarray(batch)/255

		# color images: should name labels ? or ground truth?
		labels = np.asarray(labels)/255

		input_data = batch
		ground_truth = labels
		return input_data, ground_truth, images













