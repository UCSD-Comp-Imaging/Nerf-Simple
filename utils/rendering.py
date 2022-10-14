import numpy as np
import torch 
import cv2 
from tqdm import tqdm

def render_nerf(rays, net, N, tn=2, tf=6, gpu=True):
	""" stratified sampling on a set of rays using Nerf model
	Args:
		rays (torch Tensor): B x 6 
		net (nerf model):
	Returns:
		 out: BxNx4	
	"""
	## input to nerf model BN x 6 
	## Nerf output - BN x 4  --> reshape BxNx4 
	## t sample dims - BxN
	B = rays.size(0)
	t_bins = torch.linspace(tn,tf,N+1)
	bin_diff = t_bins[1] - t_bins[0] 

	unif_samps = torch.rand(rays.size(0),N)
	ts = bin_diff* unif_samps + t_bins[:-1] # BxN 
	if gpu:
		ts = ts.cuda()
	origins = rays[:,:3] # Bx3
	dirs = rays[:,3:]  # Bx3

	disp = dirs.unsqueeze(-1)*ts.unsqueeze(1) # Bx1x3 * BxNx1 = Bx3xN
	
	## TODO how to rescale coordinates to -1,1 
	locs = origins.unsqueeze(-1) + disp # Bx3x1 + Bx3xN = Bx3xN 

	query_pts = torch.cat((locs, dirs.unsqueeze(-1).expand(-1,-1,N)),dim=1) # Bx6xN
	query_pts = query_pts.permute(0,2,1) # BxNx6
	query_pts = query_pts.reshape(-1,6)
	out = net.forward(query_pts)
	out = out.reshape(B,N,4)
	rgb, depth, alpha, acc, w = volume_render(out, ts)
	return rgb, depth, alpha, acc, w

	
def volume_render(nerf_outs, ts):
	""" computes color, depth and alphas along rays using NeRF outputs (section 4)
	Args:
		ts (torch tensor) BxN | B:number of rays, N: number of samples along a ray 
		nerf_outs (torch tensor)BxNx4 | RGB \sigma (4 values) for each sample along each ray 
	Returns: 
		rgb (torch tensor) Bx3
		depth (torch tensor) (B,)
		alphas (torch tensor) (B,N)
		acc (torch tensor) (B,)
		weights (torch tensor) (B,N) 
	"""

	deltas = ts[:,1:] - ts[:,:-1]
	deltas = torch.cat((deltas, 1e10*torch.ones_like(deltas[:,:1])), dim=1)
	## TODO add gaussian noise regularizer 

	alpha = 1 - torch.exp(-nerf_outs[...,3]*deltas)
	T = torch.exp(-torch.cumprod(nerf_outs[...,3]*deltas, dim=1))
	
	## Eqn 5 of paper w: BxN
	weights = alpha*T 
	
	## accumulated rgb color along each ray (Bx3)
	rgb = torch.sum(weights.unsqueeze(-1)*nerf_outs[...,:3], axis=1)

	## depth along each ray (B,), weighted average of samples 
	depth = torch.sum(weights * ts, axis=-1)

	## accumulation map (B,): average of weight values along a ray 
	acc = torch.sum(weights, axis=-1)
	rgb += 1-acc.unsqueeze(-1)
	return rgb, depth, alpha, acc, weights


def render_image(net, rg, batch_size=64000, im_idx=0, im_set='val'):
	""" render an image and depth map from val set (currently hardcoded) from trained NeRF model """

	gt_img = rg.samples[im_set][im_idx]['img']
	NUM_RAYS = 640000 # number of rays in image, currently hardcoded
	net = net.cuda()
	rays = rg.rays_dataset[im_set][:,im_idx*NUM_RAYS:(im_idx+1)*NUM_RAYS].transpose(1,0)
	# rays = rg.rays_dataset[im_set][im_idx*NUM_RAYS:(im_idx+1)*NUM_RAYS,:]
	rgbs = [] 
	depths = [] 
	with torch.no_grad():
		for i in tqdm(range(rays.size(0) // batch_size)):
			inp_rays = rays[i*batch_size:(i+1)*batch_size]
			rgb, depth, _, _, _ = render_nerf(inp_rays.cuda(), net, N=128)
			rgbs.append(rgb)
			depths.append(depth)

	rgb = torch.cat(rgbs).cpu()
	depth = torch.cat(depths).cpu()
	
	rgb_img = rgb.reshape(1,800,800,3) ## permuting for tensorboard
	depth_img = depth.reshape(1,800,800,1) ## permuting for tensorboard
	gt_img = gt_img.reshape(1,800,800,3)
	return rgb_img, depth_img, gt_img
		