import cv2
import numpy as np
import torch
import os 
import time
from utils.xyz import rays_single_cam
from utils.rendering import render_nerf
from tqdm import tqdm

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

def poses_to_render(r, theta, n_phi=40):
	""" list of cam poses (torch.Tensor) at radisus r, altitude theta and n_phi continuous azimuths
	Args:
		r: camera distance from origin 
		theta (degrees): altitude of camera (zenith)
	"""
	phis = np.linspace(0, 360.0, n_phi)
	poses = [torch.from_numpy(spherical_to_pose(r, theta, phi)).float() for phi in phis]
	return poses

def render_poses(net, poses, cam_params, batch_size, savepath=''):
	""" render an image and depth map from a given set of poses 
	Args:
		net: instance of Nerf() model loaded with params 
		poses: list of 4x4 (np.array) camera poses in which scene is to be rendered
		cam_params: [H, W, f]
	Returns:
		An animation of poses 
	"""
	H,W = cam_params[0], cam_params[1]
	num_imgs = len(poses)
	NUM_RAYS = H*W # number of rays in image
	net = net.cuda()
	rays_1_cam = rays_single_cam(cam_params)
	transf_mats = torch.stack(poses)
	rays_dataset = torch.matmul(transf_mats[:,:3,:3], rays_1_cam)
	cam_origins = transf_mats[:,:3,3:]
	cam_origins = cam_origins.expand(len(poses),3,H*W) #Bx3xHW
	rays_all = torch.cat((cam_origins, rays_dataset),dim=1).permute(0,2,1).reshape(-1, 6) # BHW x 6, number of cameras 

	rgb_imgs = []
	depth_imgs = []
	with torch.no_grad():
		for idx in range(num_imgs):
			rgbs = [] 
			depths = [] 
			rays = rays_all[NUM_RAYS*idx:NUM_RAYS*(idx+1), :]
			for i in tqdm(range(rays.size(0) // batch_size)):
				inp_rays = rays[i*batch_size:(i+1)*batch_size]
				rgb, depth, _, _, _ = render_nerf(inp_rays.cuda(), net, N=128)
				rgb = torch.clip(rgb, torch.tensor(0.).cuda(), torch.tensor(1.).cuda())
				rgbs.append(rgb)
				depths.append(depth)

			rgb = torch.cat(rgbs).cpu().reshape(H,W,3).numpy()
			depth = torch.cat(depths).cpu().reshape(H,W).numpy()
			rgb_imgs.append(rgb)
			depth_imgs.append(depth)
	
	tstamp = str(time.time())
	out = cv2.VideoWriter(os.path.join(savepath,f'nerf_rgb{tstamp[-10:]}.mp4'),cv2.VideoWriter_fourcc('m','p','4','v'),15, (H,W))
	for i in range(len(rgb_imgs)):
		rgb_img = cv2.cvtColor(rgb_imgs[i], cv2.COLOR_RGB2BGR)
		out.write((rgb_img*255).astype(np.uint8))
	out.release()

