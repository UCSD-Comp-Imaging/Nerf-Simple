import mitsuba as mi
import json
import numpy as np
mi.set_variant('cuda_ad_rgb')
from mitsuba import ScalarTransform4f as T
import drjit as dr
import os 
import cv2 
import tqdm 
from utils.mitsuba_utils import gen_mitsuba_data, gen_mitsuba_phaseoptic_data
import argparse 
import yaml 

## generate training data

if __name__=="__main__":
	parser = argparse.ArgumentParser(description=' mitsuba data generator')
	parser.add_argument('--config_path', type=str, default='/home/ubuntu/NeRF_CT/datagen_configs/mitsuba_datagen.yaml',
						help='location of data for training')
	args = parser.parse_args()

	with open(args.config_path) as f:
		params = yaml.load(f, Loader=yaml.FullLoader)
	print("fixing seed")
	np.random.seed(0)
	phase_optic_params = params['phase_optic_params']
	datagen_params = params['datagen_params']
	datagen_params_nopo = params['datagen_params_nopo']

	if phase_optic_params['use_phaseoptic']:
		datagen_params['phase_optic_params'] = phase_optic_params
		print("generating train data with phase optic")
		gen_mitsuba_phaseoptic_data(**datagen_params)
	else:
		print("generating train data, rendered without phase optic")
		datagen_params['data_type'] = 'train'
		gen_mitsuba_data(**datagen_params)

	print("generating validation data, rendered without phase optic")
	gen_mitsuba_data(**datagen_params_nopo)

	print("generating test data, rendering without phase optic")
	datagen_params_nopo['data_type'] = 'test'
	gen_mitsuba_data(**datagen_params_nopo)
