import numpy as np
import os 
import glob
import json
import cv2
import re 
import torch
from natsort import natsort_keygen, ns
from utils.xyz import rays_single_cam
from utils.phaseoptic import PhaseOptic, gen_phase_optic, phase_optic_rays


def load_data(path, res_factor=1, num_imgs=-1):
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
		H,W = train_img.shape[:2]
		train_img = cv2.resize(train_img, (int(W*res_factor) , int(H*res_factor)), interpolation=cv2.INTER_AREA)
		train_samples.append({'img': train_img, 'transform':transform, 'metadata':metadata})

	## generate val samples 
	print(num_val)
	val_samples = [] 
	for i in range(num_val):
		val_img = cv2.cvtColor(cv2.imread(val_img_paths[i]), cv2.COLOR_BGR2RGB) / 255.0
		metadata = val_transform['frames'][i]
		transform = torch.from_numpy(np.array(metadata['transform_matrix'])).float()
		H,W = val_img.shape[:2]
		val_img = cv2.resize(val_img, (int(W*res_factor), int(H*res_factor)), interpolation=cv2.INTER_AREA)
		val_samples.append({'img': val_img, 'transform':transform, 'metadata':metadata})
	

	test_samples = [] 
	for i in range(num_test):
		img = cv2.cvtColor(cv2.imread(test_img_paths[i]), cv2.COLOR_BGR2RGB) / 255.0
		img_depth = cv2.cvtColor(cv2.imread(test_depth_paths[i]), cv2.COLOR_BGR2RGB) / 255.0
		img_normal = cv2.cvtColor(cv2.imread(test_normal_paths[i]), cv2.COLOR_BGR2RGB) / 255.0
		metadata = test_transform['frames'][i]
		transform = torch.from_numpy(np.array(metadata['transform_matrix'])).float()
		H,W = img.shape[:2]
		img = cv2.resize(img, (int(W*res_factor), int(H*res_factor)), interpolation=cv2.INTER_AREA)

		test_samples.append({'img': img, 'img_depth': img_depth, 'img_normal':img_normal,\
			 				 'transform':transform, 'metadata':metadata})	

	## calculate image params and focal length 
	fov = train_transform['camera_angle_x']
	H, W = img.shape[:2]
	f = W /(2 * np.tan(fov/2))
	cam_params = [H,W,f]
	

	samples = {} 
	## list of phase optic configurations
	samples['train_phase_dicts'] = []
	samples['train'] = train_samples
	samples['test'] = test_samples
	samples['val'] = val_samples
	
	if 'scene_center' in train_transform:
		samples['scene_center'] = np.array(train_transform['scene_center'])

	if 'zdir' in train_transform:
		samples['zdir'] = train_transform['zdir']
	else:
		samples['zdir'] = -1
	if 'phase_optic_params' in train_transform:
		phase_dict = {}
		## TODO change this to a for loop based on how multiple phase op configs are stored in train.json
		phase_dict['phase_optic_params'] = train_transform['phase_optic_params']
		phase_dict['phase_optic_centers'] = np.array(train_transform['phase_optic_centers'])
		phase_dict['phase_optic_radii'] = np.array(train_transform['phase_optic_radii'])
		## TODO keep appending multiple phase optic configs to this list 
		samples['train_phase_dicts'].append(phase_dict)

	return samples, cam_params   

def rays_dataset(samples, cam_params, phase_optic=None, phase_optic_data_keys=['train']):
	""" Generates rays and camera origins for train test
	    and val sets under diff camera poses
	Args:
		samples - generated from the function above 
		cam_params - [H,W,f]
		phase_optic - element of class PhaseOptic with centers and radii for phase optic
		phase_optic_data_keys - subset list of ['train', 'val', 'test']
	""" 
	keys = ['train', 'test', 'val']
	rays_1_cam = rays_single_cam(cam_params, zdir=samples['zdir'])
	if phase_optic is not None:
		rays_phaseop = phase_optic_rays(cam_params, phase_optic, spp=1, zdir=samples['zdir'])
		rays_phaseop = torch.from_numpy(rays_phaseop).t().float()
	rays = {}
	cam_origins = {}
	H, W, f = cam_params
	for k in keys:
		num_images = len(samples[k])
		transf_mats = torch.stack([s['transform'] for s in samples[k]])
		if phase_optic is not None and k in phase_optic_data_keys:
			## use phase optic rays only in train data not val and test
			origins = torch.matmul(transf_mats[:,:3,:3], rays_phaseop[:3,:]) + transf_mats[:,:3,3:]
			dirs = torch.matmul(transf_mats[:,:3,:3], rays_phaseop[3:,:])
		else:
			dirs =  torch.matmul(transf_mats[:,:3,:3], rays_1_cam)
			origins = transf_mats[:,:3,3:]
			origins = origins.expand(num_images,3,H*W) #Bx3xHW
		
		rays[k] = torch.cat((origins, dirs),dim=1).permute(0,2,1).reshape(-1, 6) # BHW x 6, number of cameras 

	return rays

	## TODO Create separate function for multipe phase optics ray generation and modify selection function
class RayGenerator:
	def __init__(self, path, res_factor=1, num_imgs=-1, test_phase_dict=None, phase_data_keys=[]):
		"""
		path: root data folder which has train,test,val folders and jsons
		half_res: train at half resolution 
		test_phase_dict: (dict) render images through a phase optic specified at inference if required
		the phase optic is created using gen_phase_optic function with `test_phase_dict` as input
		phase_data_keys: subset of ['Train', 'test', 'val']. phase optic ray tracing only for data types in this list 
		"""
		samples, cam_params = load_data(path, res_factor, num_imgs)
		self.samples = samples
		self.cam_params = cam_params
		self.phase_dict = test_phase_dict
		self.train_phase_dicts = samples['train_phase_dicts']
		## TODO add support for test phase dicts 
		if test_phase_dict is not None:
			test_phase_dict['zdir'] = samples['zdir']
			element = gen_phase_optic(cam_params, **test_phase_dict)
			self.rays_dataset = rays_dataset(self.samples, cam_params, element, ['test'])
		elif len(self.train_phase_dicts) > 0 and len(phase_data_keys)>0:
			## currently use list of length 1 only, i.e. only 1 phase optic 
			for phaseop in self.train_phase_dicts:
				centers = phaseop['phase_optic_centers']
				radii = phaseop['phase_optic_radii']
				element = PhaseOptic(centers, radii, mu=1.5)
			self.rays_dataset = rays_dataset(self.samples, cam_params, element, phase_data_keys)
			## TODO create a list of rays_datasets and append rays dataset for multiple phase optics
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
		NUM_RAYS = self.cam_params[0] * self.cam_params[1]
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


		
		



