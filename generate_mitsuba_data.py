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

phase_optic_params = {
	'num_lenses': 100,
	'optic_type': 'uniform',
	'r_range_percent' : 0.5, 
	'same_sag' : True,
	'radius_scale' : 0.1,
	'base_radius': 0.2
}

datagen_params = {
	'scene_path':'scenes/lego/scene.xml',
	'datapath':'data_mitsuba_phaseoptic_debug/lego/',
	'data_type':'train', 
	'camera_angle': 0.6911,
	'spp':48,
	'width':400,
	'height':400,
	'r_range':[1.75,1.8,1],
	'phi_range':[30,360,3],
	'theta_range':[-30, -60,1],
	'random_poses': False,
	'phase_optic_params': phase_optic_params
}

datagen_params_nopo = {
	'scene_path':'scenes/lego/scene.xml',
	'datapath':'data_mitsuba_phaseoptic_debug/lego/',
	'data_type':'val', 
	'camera_angle': 0.6911,
	'spp':32,
	'width':400,
	'height':400,
	'r_range':[1.75,1.8,1],
	'phi_range':[30,360,3],
	'theta_range':[-30, -60,1],
	'random_poses': False
}

## generate training data
print("generating train data with phase optic")
gen_mitsuba_phaseoptic_data(**datagen_params)

print("generating validation data, rendered without phase optic")
gen_mitsuba_data(**datagen_params_nopo)

print("generating test data, rendering without phase optic")
datagen_params_nopo['data_type'] = 'test'
gen_mitsuba_data(**datagen_params_nopo)
