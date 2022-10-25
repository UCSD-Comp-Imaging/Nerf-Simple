from this import d
import numpy as np
import torch 
import cv2 
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from utils.xyz import positional_encoder

def render_nerf(rays, net, N, tn=2, tf=6, gpu=True, mode='Train'):
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
	dirs = dirs / torch.norm(dirs, dim=1, keepdim=True)	
	query_pts = torch.cat((locs, dirs.unsqueeze(-1).expand(-1,-1,N)),dim=1) # Bx6xN
	query_pts = query_pts.permute(0,2,1) # BxNx6
	query_pts = query_pts.reshape(-1,6)
	out = net.forward(query_pts)
	out = out.reshape(B,N,4)
	# rgb, depth, alpha, acc, w = raw2outputs(out, ts, dirs)
	rgb_v, depth_v, alpha_v, acc_v, w_v = volume_render(out, ts, dirs, mode=mode)

	return rgb_v, depth_v, alpha_v, acc_v, w_v

def render_nerf_ref(rays, net, N, tn=2, tf=6, gpu=True, mode='Train'):
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
	
	dirs = dirs / torch.norm(dirs, dim=1, keepdim=True)
	query_pts = torch.cat((locs, dirs.unsqueeze(-1).expand(-1,-1,N)),dim=1) # Bx6xN
	query_pts = query_pts.permute(0,2,1) # BxNx6
	query_pts = query_pts.reshape(-1,6)

	posx, posd = positional_encoder(query_pts)
	query_pts = torch.cat((posx, posd),dim=1)
	out = net.forward(query_pts)
	out = out.reshape(B,N,4)
	rgb, depth, alpha, acc, w = raw2outputs(out, ts, dirs)
	rgb_v, depth_v, alpha_v, acc_v, w_v = volume_render(out, ts, dirs, mode=mode)

	return rgb, depth, alpha, acc, w
	# return rgb_v, depth_v, alpha_v, acc_v, w_v

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0.0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=nn.Softplus(): 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).cuda().expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    rgb = raw[...,:3]
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = (torch.randn(raw[...,3].shape) * raw_noise_std).cuda()

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, alpha, acc_map, weights

def volume_render(nerf_outs, ts, dirs, mode='Train'):
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
	deltas = deltas * torch.norm(dirs[...,None,:], dim=-1)
	## TODO add gaussian noise regularizer 

	sigma = nerf_outs[...,3]
	if mode == 'Train':
		noise = torch.randn(sigma.shape,device='cuda')*0.0
		alpha = 1 - torch.exp(-nn.functional.softplus(sigma + noise)*deltas)
	else:
		alpha = 1 - torch.exp(-nn.functional.softplus(sigma)*deltas) 
	weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), 1.-alpha + 1e-10], -1), -1)[:, :-1]
	# T = torch.cumprod(1.0 - alpha, dim=1)
	
	# ## Eqn 5 of paper w: BxN
	# weights = alpha*T 
	
	## accumulated rgb color along each ray (Bx3)
	# rgb = torch.sigmoid(nerf_outs[...,:3])
	rgb = torch.sum(weights.unsqueeze(-1)*nerf_outs[...,:3], axis=1)
	## depth along each ray (B,), weighted average of samples 
	depth = torch.sum(weights * ts, axis=-1)

	## accumulation map (B,): average of weight values along a ray 
	acc = torch.sum(weights, axis=-1)
	
	## computing disparity from depth 
	disp = torch.max(1e-10 * torch.ones_like(depth), depth / torch.sum(weights, dim=-1))
	disp = 1. / disp

	return rgb, disp, alpha, acc, weights


def render_image(net, rg, batch_size=64000, im_idx=0, im_set='val', nerf_type='ref'):
	""" render an image and depth map from val set (currently hardcoded) from trained NeRF model """

	gt_img = rg.samples[im_set][im_idx]['img']
	H,W = gt_img.shape[0], gt_img.shape[1]

	NUM_RAYS = H*W # number of rays in image, currently hardcoded
	net = net.cuda()
	# rays = rg.rays_dataset[im_set][:,im_idx*NUM_RAYS:(im_idx+1)*NUM_RAYS].transpose(1,0)
	rays = rg.rays_dataset[im_set][im_idx*NUM_RAYS:(im_idx+1)*NUM_RAYS,:]
	rgbs = [] 
	depths = [] 
	with torch.no_grad():
		for i in tqdm(range(rays.size(0) // batch_size)):
			inp_rays = rays[i*batch_size:(i+1)*batch_size]
			if nerf_type == 'no_ref':
				rgb, depth, _, _, _ = render_nerf(inp_rays.cuda(), net, N=128)
			else:
				rgb, depth, _, _, _ = render_nerf_ref(inp_rays.cuda(), net, N=128)	
			rgb = torch.clip(rgb, torch.tensor(0.).cuda(), torch.tensor(1.).cuda())
			rgbs.append(rgb)
			depths.append(depth)

	rgb = torch.cat(rgbs).cpu()
	depth = torch.cat(depths).cpu()
	
	rgb_img = rgb.reshape(1,H,W,3) ## permuting for tensorboard
	depth_img = depth.reshape(1,H,W,1) ## permuting for tensorboard
	gt_img = gt_img.reshape(1,H,W,3)
	return rgb_img, depth_img, gt_img