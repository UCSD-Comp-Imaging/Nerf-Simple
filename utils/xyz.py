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
		vec += [torch.sin(2**i * np.pi * x), torch.cos(2**i * np.pi * x)]
	vec = torch.cat(vec, axis=1)
	return vec

def positional_encoder(vec, Lp=10, Ld=4):
	""" applies positional encoding to input vector vec (5x1) 
	Args:
		vec (torch tensor: Nx5): x,y,z,theta,phi
		Lx: number of levels for x,y,z
		Ld: number of levels for theta, phi 
	Returns:
		posx, posd (torch tensors): N x 3*2*Lx, N x 3*2*Ld
	"""
	x,y,z,theta,phi = vec[:,0:1], vec[:,1:2], vec[:,2:3], vec[:,3:4], vec[:,4:5] 
	pos_x = gamma(x, Lp)
	pos_y = gamma(y, Lp)
	pos_z = gamma(z, Lp)
	pos_d1 = gamma(torch.sin(theta)*torch.cos(phi), Ld)
	pos_d2 = gamma(torch.sin(theta)*torch.sin(phi), Ld)
	pos_d3 = gamma(torch.cos(theta), Ld)

	posx = torch.cat([pos_x, pos_y, pos_z], axis=1)
	posd = torch.cat([pos_d1, pos_d2, pos_d3], axis=1)

	return posx, posd

def gen_poses_animation():
	""" generates a list of poses for rendering, to make video animation """

