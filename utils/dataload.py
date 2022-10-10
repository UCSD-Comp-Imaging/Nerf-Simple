import numpy as np
import os 
import glob
import json
import cv2
import re 
from natsort import natsort_keygen, ns


def load_data(path):
	"""
	Assume path has the following structure - 
	path/ -
	  test/
	  train/
	  val/
	  transforms_test.json
	  transforms_train.json
	  transforms_val.json

	Assumes that frames are ordered in the json files 

	Returns:
	  samples {'train':train, 'test': test, 'val': val}
	  cam_params [H, W, f]
	"""

	train_path = os.path.join(path, 'train')
	test_path = os.path.join(path, 'test') 
	val_path = os.path.join(path, 'val')
	
	sk = natsort_keygen(alg=ns.IGNORECASE)

	train_img_paths = glob.glob(os.path.join(train_path,'*'))
	val_img_paths = glob.glob(os.path.join(val_path,'*'))
	test_img_paths = [os.path.join(test_path,fname) for fname in os.listdir(test_path) if re.match(r"r_[0-9]+.png", fname)]
	test_depth_paths = glob.glob(os.path.join(test_path,'r_*_depth*'))
	test_normal_paths = glob.glob(os.path.join(test_path, 'r_*_normal*'))

	train_img_paths.sort(key=sk)
	val_img_paths.sort(key=sk)
	test_img_paths.sort(key=sk)
	test_depth_paths.sort(key=sk)
	test_normal_paths.sort(key=sk)
	
	with open(os.path.join(path, 'transforms_train.json')) as f:
		train_transform = json.load(f)
	with open(os.path.join(path, 'transforms_test.json')) as f:
		test_transform = json.load(f)
	with open(os.path.join(path, 'transforms_val.json')) as f:
		val_transform = json.load(f)

	num_train = len(train_img_paths)
	num_val = len(val_img_paths)
	num_test = len(test_img_paths)

	## generate training samples 
	train_samples = []
	for i in range(num_train):
		train_img = cv2.cvtColor(cv2.imread(train_img_paths[i]), cv2.COLOR_BGR2RGB) / 255.0
		transform = train_transform['frames'][i]
		transform['transform_matrix'] = np.array(transform['transform_matrix'])
		train_samples.append({'img': train_img, 'transform':transform})

	## generate val samples 
	val_samples = [] 
	for i in range(num_val):
		val_img = cv2.cvtColor(cv2.imread(val_img_paths[i]), cv2.COLOR_BGR2RGB) / 255.0
		transform = val_transform['frames'][i]
		transform['transform_matrix'] = np.array(transform['transform_matrix'])
		val_samples.append({'img': val_img, 'transform':transform})
	

	test_samples = [] 
	for i in range(num_test):
		img = cv2.cvtColor(cv2.imread(test_img_paths[i]), cv2.COLOR_BGR2RGB) / 255.0
		img_depth = cv2.cvtColor(cv2.imread(test_depth_paths[i]), cv2.COLOR_BGR2RGB) / 255.0
		img_normal = cv2.cvtColor(cv2.imread(test_normal_paths[i]), cv2.COLOR_BGR2RGB) / 255.0
		transform = test_transform['frames'][i]
		transform['transform_matrix'] = np.array(transform['transform_matrix'])
		test_samples.append({'img': img, 'img_depth': img_depth, 'img_normal':img_normal, 'transform':transform})	

	## calculate image params and focal length 
	fov = train_transform['camera_angle_x']
	H, W = img.shape[:2]
	f = W /(2 * np.tan(fov/2))
	cam_params = [H,W,f]

	## TODO: Implement half res image loading 
	samples = {} 
	samples['train'] = train_samples
	samples['test'] = test_samples
	samples['val'] = val_samples
	return samples, cam_params   



	
