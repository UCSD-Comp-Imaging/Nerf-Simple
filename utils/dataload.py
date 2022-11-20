import numpy as np
import os 
import glob
import json
import cv2
import re 
import torch
from natsort import natsort_keygen, ns
from utils.xyz import rays_single_cam
from utils.phaseoptic import PhaseOptic, unif_lenslet_params, raytrace_phaseoptic


def load_data(path, half_res=True, num_imgs=-1):
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

	if num_imgs < 0:
		num_train = len(train_img_paths)
		num_val = len(val_img_paths)
		num_test = len(test_img_paths)
	else:

		num_train = num_val = num_test = num_imgs

	## generate training samples 
	train_samples = []
	for i in range(num_train):
		train_img = cv2.cvtColor(cv2.imread(train_img_paths[i]), cv2.COLOR_BGR2RGB) / 255.0
		metadata = train_transform['frames'][i]
		transform = torch.from_numpy(np.array(metadata['transform_matrix'])).float()
		if half_res:
			H,W = train_img.shape[:2]
			train_img = cv2.resize(train_img, (W//2 , H//2 ), interpolation=cv2.INTER_AREA)
		train_samples.append({'img': train_img, 'transform':transform, 'metadata':metadata})

	## generate val samples 
	val_samples = [] 
	for i in range(num_val):
		val_img = cv2.cvtColor(cv2.imread(val_img_paths[i]), cv2.COLOR_BGR2RGB) / 255.0
		metadata = val_transform['frames'][i]
		transform = torch.from_numpy(np.array(metadata['transform_matrix'])).float()
		if half_res:
			H,W = val_img.shape[:2]
			val_img = cv2.resize(val_img, (W//2, H//2), interpolation=cv2.INTER_AREA)

		val_samples.append({'img': val_img, 'transform':transform, 'metadata':metadata})
	

	test_samples = [] 
	for i in range(num_test):
		img = cv2.cvtColor(cv2.imread(test_img_paths[i]), cv2.COLOR_BGR2RGB) / 255.0
		img_depth = cv2.cvtColor(cv2.imread(test_depth_paths[i]), cv2.COLOR_BGR2RGB) / 255.0
		img_normal = cv2.cvtColor(cv2.imread(test_normal_paths[i]), cv2.COLOR_BGR2RGB) / 255.0
		metadata = test_transform['frames'][i]
		transform = torch.from_numpy(np.array(metadata['transform_matrix'])).float()
		if half_res:
			H,W = img.shape[:2]
			img = cv2.resize(img, (W//2 ,H//2), interpolation=cv2.INTER_AREA)

		test_samples.append({'img': img, 'img_depth': img_depth, 'img_normal':img_normal,\
			 				 'transform':transform, 'metadata':metadata})	

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

def rays_dataset(samples, cam_params, phase_optic=None):
	""" Generates rays and camera origins for train test and val sets under diff camera poses""" 
	keys = ['train', 'test', 'val']
	rays_1_cam = rays_single_cam(cam_params)
	if phase_optic is not None:
		out = raytrace_phaseoptic(cam_params, phase_optic)
		_,_, rays_phaseop = out['rays_trace']
		rays_phaseop = torch.from_numpy(rays_phaseop).t().float()
	rays = {}
	cam_origins = {}
	H, W, f = cam_params
	for k in keys:
		num_images = len(samples[k])
		transf_mats = torch.stack([s['transform'] for s in samples[k]])
		if phase_optic is None:
			dirs =  torch.matmul(transf_mats[:,:3,:3], rays_1_cam)
			origins = transf_mats[:,:3,3:]
			origins = origins.expand(num_images,3,H*W) #Bx3xHW
		else:
			origins = torch.matmul(transf_mats[:,:3,:3], rays_phaseop[:3,:]) + transf_mats[:,:3,3:]
			dirs = torch.matmul(transf_mats[:,:3,:3], rays_phaseop[3:,:])
		rays[k] = torch.cat((origins, dirs),dim=1).permute(0,2,1).reshape(-1, 6) # BHW x 6, number of cameras 

	return rays

class RayGenerator:
	def __init__(self, path, half_res=True, num_imgs=-1, phase_dict=None):
		samples, cam_params = load_data(path, half_res, num_imgs)
		self.samples = samples
		self.cam_params = cam_params
		self.H = cam_params[0]
		self.W = cam_params[1]
		self.f = cam_params[2]
		self.phase_dict = phase_dict
		if phase_dict is not None and self.phase_dict['use_phase_optic']:
			num_lenses = phase_dict['num_lenses']
			radius_scale = phase_dict['radius_scale']
			## currently only uniform lenslets supported 
			centers, radii = unif_lenslet_params(num_lenses,cam_params,radius_scale)
			## generating max over 
			phase_optic = PhaseOptic(centers, radii, mu=1.5)
			self.rays_dataset = rays_dataset(self.samples, cam_params, phase_optic)
		else:
			self.rays_dataset = rays_dataset(self.samples, cam_params)

	def select(self, mode='train', N=4096):
		""" randomly selects N train/test/val rays
		Args:
			mode: 'train', 'test', 'val'
			N: number of rays to sample 
		Returns:
			rays (torch Tensor): Nx6 
			ray_ids: Nx1 
		"""
		data = self.rays_dataset[mode]
		ray_ids = torch.randperm(data.size(0))[:N]
		rays = data[ray_ids,:]
		return rays, ray_ids

	def select_imgs(self, mode='train', N=4096, im_idxs=[0,1,2]):
		""" randomly selects N train/test/val rays from a given image
		Args:
			mode: 'train', 'test', 'val'
			N: number of rays to sample
			im_idxs: which image to select
		Returns:
			rays (torch Tensor): Nx6 
			ray_ids: Nx1 
		"""
		NUM_RAYS = self.H * self.W
		data = []
		rays_idxs = [] 
		for im_idx in im_idxs:
			data.append(self.rays_dataset[mode][im_idx*NUM_RAYS:(im_idx + 1)*NUM_RAYS,:])
			rays_idxs.append(np.arange(im_idx*NUM_RAYS, (im_idx + 1)*NUM_RAYS))
		data = torch.cat(data, dim=0)	

		samples = self.samples[mode]
		select_ids = np.random.choice(data.size(0), (N,), replace=False)
		rays_idxs = np.concatenate(rays_idxs)
		rays = data[select_ids, :]
		ray_ids = rays_idxs[select_ids]

		return rays, ray_ids


		
		



