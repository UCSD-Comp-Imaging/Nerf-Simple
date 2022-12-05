import os 
import time 
import torch 
import torch.nn as nn
import cv2 
import numpy as np
from utils.dataload import load_data, RayGenerator
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from utils.nets import Nerf
from utils.xyz import * 
from utils.rendering import *
# from utils.visualization import render_poses, poses_to_render
import argparse 
import yaml 
from tqdm import tqdm 
from torchmetrics import StructuralSimilarityIndexMeasure


def img_mse(gt, pred):
	if not torch.is_tensor(gt):
		gt = torch.from_numpy(gt).float()
	return torch.mean((pred - gt)**2)

def img_psnr(gt, pred):
	ten = torch.tensor(10)
	if not torch.is_tensor(gt):
		gt = torch.from_numpy(gt).float()
	psnr = 20*torch.log(torch.max(gt))/torch.log(ten) - 10 * torch.log(img_mse(gt, pred))/torch.log(ten)
	return psnr 

def img_ssim(gt, pred):
	if not torch.is_tensor(gt):	
		gt =torch.from_numpy(gt).float()
	ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
	score = ssim(pred.permute(0,3,1,2), gt.permute(0,3,1,2))
	return score

def test(params):
	assert os.path.exists(params['loadpath']), "model path doesn't exist"
	if not os.path.exists(os.path.join(params['savepath'], params['exp_name'])):
		os.makedirs(os.path.join(params['savepath'], params['exp_name']))
	
	savepath = os.path.join(params['savepath'], params['exp_name'])
	batch_size = params['batch_size']
	rg = RayGenerator(params['datapath'], params['res_factor'])
	cam_params = rg.cam_params
	Lp = params.get('Lp', 10)
	Ld = params.get('Ld', 4)
	hidden = params.get('H', 256)
	print(Lp, Ld, hidden)
	net = Nerf(Lp, Ld, hidden).cuda()
	# net = Nerf().cuda()
	net.load_state_dict(torch.load(params['loadpath']),strict=True)
	
	if params['animation']:
		theta = -params['theta']
		n_phi = params['num_poses']
		poses = poses_to_render(r=params.get('r', 4), theta=theta, n_phi=n_phi,
								mitsuba_pose=params.get('mitsuba_pose', False),
								scene_center=rg.samples.get('scene_center', np.zeros(3)))
		render_poses(net, poses, cam_params, batch_size, savepath,\
					N=params['Nf'], tn=params['tn'], tf=params['tf'], zdir=rg.samples.get('zdir', -1))
		return

	if params['cone_animation']:
		print("cone animation")
		cone_idx = params['cone_idx']
		im_set = params['cone_im_set']
		cone_params = rg.samples[im_set][cone_idx]['metadata']
		r, theta, phi = cone_params['r'], cone_params['theta'], cone_params['phi'] 
		poses = cone_poses_to_render(r, theta, phi, rg.samples['scene_center'],
								   params['half_angle'], params['num_cone_pts'])
		render_poses(net, poses, cam_params, batch_size, savepath,\
					N=params['Nf'], tn=params['tn'], tf=params['tf'], zdir=rg.samples.get('zdir', -1))
		return 

	im_set = params['im_set'] # can be 'train', 'test', 'val' depending on whcich images to render
	print(f"saving images to {params['savepath']}")
	idxs = params['im_idxs'] if type(params['im_idxs']) is list else list(range(len(rg.samples[im_set])))
	avg_psnr = 0
	avg_ssim = 0
	with torch.no_grad():
		for idx in idxs:
			rgb_img, depth_img, gt_img = render_image(net, rg, batch_size=params['batch_size'], \
											 		  im_idx=idx, im_set=im_set, N=params['Nf'],tn= params['tn'], tf=params['tf'])
			avg_psnr += img_psnr(gt_img, rgb_img)
			avg_ssim += img_ssim(gt_img, rgb_img)
			rgb_out = torch.cat((torch.from_numpy(gt_img), rgb_img),axis=0)
			save_image(make_grid(rgb_out.permute(0,3,1,2)), os.path.join(params['savepath'], params['exp_name'], f'rgb_{idx}.png'))
			save_image(make_grid(depth_img.permute(0,3,1,2)), os.path.join(params['savepath'], params['exp_name'], f'depth_{idx}.png'))
	avg_psnr = avg_psnr / len(idxs)
	avg_ssim = avg_ssim / len(idxs)
	print(f"Average PSNR : {avg_psnr}")
	print(f"Average SSIM : {avg_ssim}")
if __name__=="__main__":
	parser = argparse.ArgumentParser(description='NeRF scene')
	parser.add_argument('--config_path', type=str, default='/home/ubuntu/NeRF_CT/configs/lego.yaml',
						help='location of data for training')
	args = parser.parse_args()

	with open(args.config_path) as f:
		params = yaml.load(f, Loader=yaml.FullLoader)
	test(params['test_params'])


