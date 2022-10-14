import os 
import time 
import torch 
import torch.nn as nn
import cv2 
import numpy as np
from utils.dataload import load_data, RayGenerator
from torch.utils.tensorboard import SummaryWriter
from utils.nets import Nerf
from utils.xyz import * 
from utils.rendering import *
import argparse 
import yaml 
from tqdm import tqdm 


def train(params):
	if not os.path.exists(params['savepath']):
		os.makedirs(params['savepath'])
	writer = SummaryWriter('logs/run_{}/'.format(str(time.time())[-10:]))
	batch_size = params['batch_size']
	rg = RayGenerator(params['datapath'])
	train_imgs = torch.stack([torch.from_numpy(s['img']) for s in rg.samples['train']]).reshape(-1,3)
	# val_imgs = torch.stack([torch.from_numpy(s['img']) for s in rg.samples['val']]).reshape(-1,3)

	net = Nerf().cuda()
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
	## TODO: Add code to load state dict from pre-trained model 
	for i in tqdm(range(params['num_iters'])):
		rays, ray_ids = rg.select(mode='train', N=batch_size)
		gt_colors = train_imgs[ray_ids,:].float().cuda()
		optimizer.zero_grad()
		rgb, depth, alpha, acc, w = render_nerf(rays.cuda(), net, params['Nf'])
		loss = criterion(rgb, gt_colors)
		if i % params['ckpt_loss'] == 0:
			writer.add_scalar("Loss/train", loss.item(), i+1)
			print(f'loss: {loss.item()} | epoch: {i+1} ')
		if i % params['ckpt_images'] == 0:
			print("--- rendering image ---")
			rgb_img, depth_img, gt_img = render_image(net, rg, batch_size=12800,\
											 		  im_idx=0, im_set='val')
			writer.add_images('Val/RGB', rgb_img, global_step=i+1, dataformats='NHWC')
			writer.add_images('Val/Depth', depth_img, global_step=i+1, dataformats='NHWC')
			writer.add_images('Val/GT', gt_img, global_step=i+1, dataformats='NHWC')
		
		if i% params['ckpt_model'] == 0:
			print("saving model")
			tstamp = str(time.time())
			torch.save(net.state_dict(), os.path.join(params['savepath'], tstamp+'.pth'))

		loss.backward()
		optimizer.step()
	
	print("saving final model")
	tstamp = str(time.time())
	torch.save(net.state_dict(), os.path.join(params['savepath'], tstamp+'.pth'))

	
if __name__=="__main__":
	parser = argparse.ArgumentParser(description='NeRF scene')
	parser.add_argument('--config_path', type=str, default='/home/ubuntu/NeRF_CT/configs/lego.yaml',
						help='location of data for training')
	args = parser.parse_args()

	with open(args.config_path) as f:
		params = yaml.load(f, Loader=yaml.FullLoader)
	train(params)


