import cv2
import numpy as np
import torch
import math
import warnings
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
from mitsuba import ScalarTransform4f as T

def gamma(x, L=4):	
	assert torch.is_tensor(x), "input needs to be a torch tensor"
	if torch.any(x < -1) or torch.any(x > 1):
		warnings.warn("input not in range -1,1, check rescaling")
	vec = []
	for i in range(L):
		vec += [torch.sin(2**i * x), torch.cos(2**i * x)]
	if L == 0:
		vec = torch.empty(0)
	else:
		vec = torch.cat(vec, axis=1)
	return vec

def positional_encoder(vec, Lp=10, Ld=4):
	""" applies positional encoding to input vector vec (5x1) 
	Args:
		vec (torch tensor: Bx6): x, y, z, d1, d2, d3
		Lx: number of levels for x, y, z
		Ld: number of levels for d1, d2, d3 
	Returns:
		posx, posd (torch tensors): B x 3*2*Lx, B x 3*2*Ld
	"""
	x,y,z,d1,d2,d3 = vec[:,0:1], vec[:,1:2], vec[:,2:3], vec[:,3:4], vec[:,4:5], vec[:,5:6] 
	pos_x = gamma(x, Lp)
	pos_y = gamma(y, Lp)
	pos_z = gamma(z, Lp)
	pos_d1 = gamma(d1, Ld)
	pos_d2 = gamma(d2, Ld)
	pos_d3 = gamma(d3, Ld)

	posx = torch.cat([x,y,z], axis=1) if Lp ==0 else torch.cat([x, y, z, pos_x, pos_y, pos_z], axis=1)
	posd =  torch.cat([d1,d2,d3], axis=1)if Ld ==0 else torch.cat([d1, d2, d3, pos_d1, pos_d2, pos_d3], axis=1)

	return posx, posd

def rays_single_cam(cam_params, spp=1, zdir=-1):
	""" takes in camera params H,W,f returns H*W ray directions with origin 0,0,0
	Args:
		cam_params (list): [H, W, f]
		spp (samples per pixel, int): if > 1 sample n samples randomly in each pixel
		For NeRF , spp is always assumed to be 1 
		mitsuba (bool) : If true, camera looks at +ve z
	Returns:
		rays (torch Tensor): 3 x HW
	"""

	H , W, f  = cam_params
	Hl = torch.arange(H) - H//2
	Wl = torch.arange(W) - W//2

	grid_x, grid_y = torch.meshgrid(Wl, Hl)
	grid_x = grid_x.t()
	grid_y = grid_y.t()
	if spp > 1:
		unif_samps = torch.rand(H,W,spp*2)*(Hl[1] - Hl[0])
		grid_x_spp = unif_samps[:,:,:spp] + grid_x.unsqueeze(-1)
		grid_y_spp = unif_samps[:,:,spp:] + grid_y.unsqueeze(-1)

		rays_spp = torch.stack((-zdir*grid_x_spp/f, -grid_y_spp/f, zdir*torch.ones_like(grid_x_spp))).float()
		# rays_spp = rays_spp.permute(0,2,1,3)
		rays_spp = torch.reshape(rays_spp, (3,-1))
		return rays_spp

	rays = torch.stack((-zdir*grid_x/f, -grid_y/f, zdir*torch.ones_like(grid_x))).float()
	# rays = rays.permute(0,2,1)
	rays = torch.reshape(rays, (3,-1)) # 640K ray directions (if H,W = 800 and spp=1 else 640K*n)
	return rays

def polar_to_mat(theta):
	""" use r, theta phi to generate transformation matrix for generating pose of camera w.r.t world """
	theta_mat = np.array([[1., 0., 0., 0.],
						  [0., np.cos(theta), np.sin(theta), 0.],
						  [0., -np.sin(theta), np.cos(theta), 0.],
						  [0., 0., 0., 1.]])
	return theta_mat 

def phi_to_mat(phi):
	mat = np.array([[np.cos(phi), np.sin(phi), 0., 0.],
					[-np.sin(phi), np.cos(phi), 0., 0.],
					[0., 0., 1., 0],
					[0., 0., 0., 1. ]])
	return mat 

def spherical_to_pose(r, theta, phi):
	""" returns pose corresponding to the input spherical coordinates r, theta, phi of the camera """
	theta = np.radians(theta)
	phi = np.radians(phi)
	trans_mat = np.array([[1., 0., 0., 0.],
						  [0., 1., 0., 0.],
						  [0., 0., 1., r],
						  [0., 0., 0., 1.]])
	theta_mat = polar_to_mat(theta)
	phi_mat = phi_to_mat(phi)
	pose = phi_mat@theta_mat@trans_mat
	return pose

def poses_to_render(r, theta, n_phi=40, mitsuba_pose=False, scene_center=np.zeros(3)):
	""" list of cam poses (torch.Tensor) at radisus r, altitude theta and n_phi continuous azimuths
	Args:
		r: camera distance from origin 
		theta (degrees): altitude of camera (zenith)
		n_phi: number of poses in phi direction 
		scene_center - used for mitsuba scenes
	"""
	phis = np.linspace(0, 360.0, n_phi)
	if mitsuba_pose:
		poses = [torch.from_numpy(mitsuba_spherical_to_pose(r, theta, phi, scene_center)).float() for phi in phis]	
	else:
		poses = [torch.from_numpy(spherical_to_pose(r, theta, phi)).float() for phi in phis]
	return poses

def mitsuba_spherical_to_pose(r, theta, phi, scene_center):
	""" function to generate rendering trained nerf in mitsuba coordinate system """
	origin = T.rotate([0, 1, 0], phi).rotate([1, 0, 0], -np.abs(theta))@mi.ScalarPoint3f([0, 0, r])
	target = np.array(scene_center)
	origin = origin + target 
	cam_pose = T.translate(target).look_at(
			origin=origin,
			target=[0, 0, 0],
			up=[0, 1, 0]
		)
	pose_r = np.array(cam_pose.matrix)
	return pose_r

def transform_rays(rays, pose):
	""" transform rays (np.array Nx6) using given pose transformation
	
	Args:
		rays (np.array) Nx6  :3 origin, 3: directions 
		pose (np.array) 4x4 
	Returns 
		rays_transformed (np.array) Nx6 
	"""
	## applying sensor pose to generated rays
	origins = rays[:,:3]
	dirs = rays[:,3:] 
	dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
	origins_h = np.hstack([origins, np.ones((rays.shape[0],1))])
	new_origins = (pose @ origins_h.T).T[:,:3]
	new_dirs = dirs @ pose[:3,:3].T
	rays_transformed = np.concatenate((new_origins, new_dirs),axis=1)
	return rays_transformed 