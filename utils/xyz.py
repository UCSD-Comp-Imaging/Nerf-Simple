import cv2
import numpy as np
import torch
import warnings

def gamma(x, L=4):	
	assert torch.is_tensor(x), "input needs to be a torch tensor"
	if torch.any(x < -1) or torch.any(x > 1):
		warnings.warn("input not in range -1,1, check rescaling")
	vec = []
	for i in range(L):
		vec += [torch.sin(2**i * x), torch.cos(2**i * x)]
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

	posx = torch.cat([x, y, z, pos_x, pos_y, pos_z], axis=1)
	posd = torch.cat([d1, d2, d3, pos_d1, pos_d2, pos_d3], axis=1)

	return posx, posd

def rays_single_cam(cam_params):
	""" takes in camera params H,W,f returns H*W ray directions with origin 0,0,0
	Args:
		cam_params (list): [H, W, f]
	Returns:
		rays (torch Tensor): 3 x HW
	"""
	H , W, f  = cam_params
	Hl = torch.arange(H) - H//2
	Wl = torch.arange(W) - W//2
	grid_x, grid_y = torch.meshgrid(Wl, Hl)
	rays = torch.stack((grid_x/f, -grid_y/f, -1*torch.ones_like(grid_x))).float()
	# rays = rays / torch.norm(rays, dim=0)
	rays = rays.permute(0,2,1)
	rays = torch.reshape(rays, (3,-1)) # 640K ray directions (if H,W = 800), normalized
	return rays

def gen_poses_animation():
	""" generates a list of poses for rendering, to make video animation """
	pass
