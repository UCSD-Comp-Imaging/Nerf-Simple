import mitsuba as mi
import json
import numpy as np
mi.set_variant('cuda_ad_rgb')
from mitsuba import ScalarTransform4f as T
import drjit as dr
import os 
import cv2 
import tqdm 
from utils.mitsuba_utils import gen_sensor, visualize_integrator, gen_mitsuba_data, gen_mitsuba_phaseoptic_data

phase_optic_params = {
	'num_lenses': 100,
	'optic_type': 'random',
	'r_range_percent' : 0.25, 
	'same_sag' : True,
	'radius_scale' : 1.5
}

datagen_params = {
	'scene_path':'scenes/lego/scene.xml',
	'datapath':'data_mitsuba_phaseoptic/lego/',
	'data_type':'train', 
	'camera_angle': 0.6911,
	'spp':32,
	'width':400,
	'height':400,
	'r_range':[1.75,1.8,1],
	'phi_range':[60,360,20],
	'theta_range':[-30, -60,1],
	'phase_optic_params': phase_optic_params
}

datagen_params_nopo = {
	'scene_path':'scenes/lego/scene.xml',
	'datapath':'data_mitsuba_nopo/lego/',
	'data_type':'val', 
	'camera_angle': 0.6911,
	'spp':32,
	'width':400,
	'height':400,
	'r_range':[1.75,1.8,1],
	'phi_range':[30,360,20],
	'theta_range':[-30, -60,1]
}

gen_mitsuba_data(**datagen_params_nopo)

imgs = gen_mitsuba_phaseoptic_data(**datagen_params)