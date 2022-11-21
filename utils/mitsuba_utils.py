import mitsuba
import drjit
from typing import Union
import mitsuba as mi 
import matplotlib.pyplot as plt
import os 
import cv2
import numpy as np 
import drjit as dr
import gc
import json
from tqdm import tqdm
mi.set_variant('cuda_ad_rgb')

from mitsuba import ScalarTransform4f as T
import numpy as np 
from utils.phaseoptic import unif_lenslet_params, random_lenslet_params, raytrace_phaseoptic, PhaseOptic, visu3d_tracer


def gen_sensor(r:float, phi: float, theta: float, fov:float, spp=256, width=800, height=800):
	"""
	fov (float): fov of camera in degrees
	pose (mitsuba ScalarTransform4f object): pose of camera / sensor
	source orign is hardcoded, beaware
	"""
	# Apply two rotations to convert from spherical coordinates to world 3D coordinates.
	origin = T.rotate([0, 1, 0], phi).rotate([0, 0, 1], theta)@mi.ScalarPoint3f([0, r, 0])
	target = np.array([0.4, 0.45, 0.5])
	origin = origin + target 
	cam_pose = T.look_at(
			origin=origin,
			target=[0.4, 0.45, 0.5],
			up=[0, 1, 0]
		)

	origin = np.array(origin)
	target = np.array([0.4, 0.45, 0.5])
	dir = (target - origin)/np.linalg.norm(origin-target)
	up = np.array([0,1,0])

	right = np.cross(dir, up) / np.linalg.norm(np.cross(up, dir))
	up = np.cross(right, dir)


	pose_r = np.eye(4)
	pose_r[:3,:3] = np.vstack([right.T, up.T, -dir.T]).T
	pose_r[:3,3] = origin

	out_dict = {}
	out_dict['sensor'] = mi.load_dict({
		'type': 'perspective',
		'fov': fov,
		'to_world': cam_pose,
		'sampler': {
			'type': 'independent',
			'sample_count': spp
		},
		'film': {
			'type': 'hdrfilm',
			'width': width,
			'height': height,
			'rfilter': {
				'type': 'gaussian',
			},
			'pixel_format': 'rgb',
		},
	})
	out_dict['sensor_pose'] = pose_r
	return out_dict

## Generate Training data 

def gen_sensors(r_range, phi_range, theta_range, fov, spp, width, height):
	"""
	fov (float) : field of view in degree 
	spp (int) : sample count for renderer
	width (int) : width of image to generate
	height (int) : height of image to generate
	phi_range (list) : [phi_min, phi_max, num_phi] to simulate num_phi values between phi_min and phi_max
	theta_range (list) : [theta_min, theta_max, num_theta] 
	r_range (list) : [r_min, r_max, num_r] to simulate
	"""
	# r, theta, phi = 3, -40, 0
	phis = np.random.uniform(*phi_range)
	rs = np.random.uniform(*r_range)
	thetas = np.random.uniform(*theta_range)
	
	sensor_dicts = []
	
	for r in rs:
		for theta in thetas:
			for phi in phis:
				sensor_dict = gen_sensor(r, phi, theta, fov, spp, width, height)
				sensor_dicts.append(sensor_dict)

	return sensor_dicts

def gen_mitsuba_data(scene_path, datapath, data_type, camera_angle, \
                     spp=32, width=800, height=800, r_range=[1.75,1.8, 1], \
					 phi_range=[30,360, 20], theta_range=[-30,-60, 1]):
	""" Generate standard mitsuba data without phase optic and save transforms_{data_type}.json 
	scene_path (str): path to scene.xml file 
	datapath (str): path where to store data
	data_type (str): 'train', 'test', 'val' 
	camera_angle (float) : fov in radian 
	spp (int) : sample count for renderer
	width (int) : width of image to generate
	height (int) : height of image to generate
	r_range (list) : [r_min, r_max, num_r] to simulate
	phi_range (list) : [phi_min, phi_max, num_phi] to simulate num_phi values between phi_min and phi_max
	theta_range (list) : [theta_min, theta_max, num_theta] 
	"""
	dn_integrator = mi.load_dict({'type':'aov', 'aovs': 'dd.z:depth,nn:sh_normal'})
	train_mitsuba_json = {}
	train_mitsuba_json['camera_angle_x'] = camera_angle
	train_mitsuba_json['frames']= [] 
	savepath = os.path.join(datapath, data_type)
	if not os.path.exists(savepath):
		os.makedirs(savepath)

	scene = mi.load_file(scene_path)
	fov = camera_angle*180/(np.pi)
	img = mi.render(scene)
	sensor_dicts = gen_sensors(r_range, phi_range, theta_range, fov, spp, width, height)
	
	for i,sdict in tqdm(enumerate(sensor_dicts)):
		sensor = sdict['sensor']
		sensor_pose = sdict['sensor_pose'].tolist()
		image = mi.render(scene, sensor=sensor, spp=256)
		depth_normal_image = mi.render(scene, sensor=sensor, integrator=dn_integrator, spp=256)
		mi.util.write_bitmap(os.path.join(savepath, f'r_{i}.png'), image)
		if data_type == 'test':
			mi.util.write_bitmap(os.path.join(savepath, f'r_{i}_depth.png'), depth_normal_image[:,:,3])
			mi.util.write_bitmap(os.path.join(savepath, f'r_{i}_normal.png'), depth_normal_image[:,:,4:])
		train_mitsuba_json['frames'].append(
			{
				"file_path": f"./train/r_{i}",
				"rotation": 0.012566370614359171, ## not useful .
				"transform_matrix": sensor_pose
			}
		)
	json.dump(train_mitsuba_json, open(os.path.join(datapath, f'transforms_{data_type}.json'),'w'))


def gen_phase_optic(cam_params, num_lenses, optic_type='uniform',
 					r_range_percent=0.5, same_sag=True, radius_scale=1.5):
	""" function to generate a phase optic element 
	Args:
		cam_params (list): [H,W,f]
		num_lenses (int) : total number of microlens elements on phase optic
		optic_type (str): 'uniform', 'random'
		radius_scale (float): factor with which to scale radius
		r_range_percent (float): percentage value between 0 and 1. randomly samples radii between B(1 +- radius_range).
		B is base radius computed based on number of lenslets, and sensor size. 
		same_sag (bool): keep the sag on the lens array same for all lenslets, by adjusting lens centers 
	returns:
		element (PhaseOptic) : instance of class PhaseOptic 
	"""
	if optic_type == 'uniform':
		centers, radii = unif_lenslet_params(num_lenses,cam_params,radius_scale)
	if optic_type == 'random':
		centers, radii = random_lenslet_params(num_lenses, cam_params, radius_scale, r_range_percent, same_sag) 
	element = PhaseOptic(centers, radii, mu=1.5)
	
	return element 

def phase_optic_rays(cam_params, phase_optic, spp=32):
	""" function to generate rays exiting a phase optic in canonical space
	Args:
		cam_params (list) [H, W, f]
		phase optic (PhaseOptic) 
		spp : samples per pixel  
	Returns:
		 
	"""
	out = raytrace_phaseoptic(cam_params, phase_optic, spp=spp)
	r0, r2, r4 = out['rays_trace']
	valid = out['valid_intersects']
	print(f"Percentage of valid intersects: {100*valid.shape[0]/r0.shape[0]} %")
	print("-------------")

	return r4 

def transform_rays(rays, pose):
	""" transform rays (np.array Nx6) using given pose transformation
	
	Args:
		rays (np.array) Nx6  :3 origin, 3: directions 
		pose (np.array) 4x4 
	Returns 
		rays_transformed (np.array) Nx6 
	"""
	## applying sensor pose to generated rays
	origins = rays[:,:3]
	dirs = rays[:,3:] 
	dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
	origins_h = np.hstack([origins, np.ones((rays.shape[0],1))])
	new_origins = (pose @ origins_h.T).T[:,:3]
	new_dirs = dirs @ pose[:3,:3].T
	rays_transformed = np.concatenate((new_origins, new_dirs),axis=1)
	return rays_transformed 

def gen_mitsuba_phaseoptic_data(scene_path, datapath, data_type, \
								camera_angle, spp=32, width=800, \
								height=800, r_range=[1.75,1.8, 1], \
								phi_range=[30,360, 20], theta_range=[-30,-60, 1], \
								phase_optic_params=None):
	"""
	render data through a phase optic from multiple sensors. 
	phase_optic_params (dict) with keys - num_lenses, optic_type,
	r_range_percent, same_sag, radius_scale
	"""
	imgs = []
	if phase_optic_params is None:
		raise Exception(" no phase optic parameters specified")
	mitsuba_json = {}
	mitsuba_json['camera_angle_x'] = camera_angle
	mitsuba_json['frames']= [] 
	savepath = os.path.join(datapath, data_type)
	if not os.path.exists(savepath):
		os.makedirs(savepath)

	scene = mi.load_file(scene_path)
	img = mi.render(scene)
	fov = camera_angle*180/(np.pi)
	f = width /(2 * np.tan(camera_angle/2))
	cam_params = [height, width, f]
	phase_optic_params['cam_params'] = cam_params
	phase_optic = gen_phase_optic(**phase_optic_params)
	rays_phase_optic = phase_optic_rays(cam_params, phase_optic, spp)
	## generating rays exiting the phase optic
	sensor_dicts = gen_sensors(r_range, phi_range, theta_range, fov, spp, width, height)
	mode = dr.ADMode.Primal
	integrator = mi.load_dict({'type':'prb',
						   'max_depth': 8,
						   'hide_emitters': True})
	
	for i,sdict in tqdm(enumerate(sensor_dicts)):
		sensor = sdict['sensor']
		sensor_pose = sdict['sensor_pose']
		rays_po = transform_rays(rays_phase_optic, sensor_pose)
		rays_po = mi.Ray3f(o = rays_po[:,:3], d= rays_po[:,3:])

		img = render_rays(integrator, scene, rays_po, sensor, spp=spp)
		mi.util.write_bitmap(os.path.join(savepath, f'r_{i}.png'),img)
		mitsuba_json['frames'].append(
			{
				"file_path": f"./train/r_{i}",
				"rotation": 0.012566370614359171, ## not useful .
				"transform_matrix": sensor_pose.tolist(),
			}
		)
	mitsuba_json['phase_optic_centers'] = phase_optic.centers.tolist()
	mitsuba_json['phase_optic_radii'] = phase_optic.radii.tolist()
	json.dump(mitsuba_json, open(os.path.join(datapath, f'transforms_{data_type}.json'),'w'))
	return imgs

def integrator_to_np(L:mitsuba.cuda_ad_rgb.Color3f, active:drjit.cuda.ad.Bool, cam_params:list):
	H,W,f = cam_params
	NUM_PIX = H*W
	L_np = np.array(L)
	L_np_ = L_np.reshape(NUM_PIX,-1,3)

	active_np = np.array(active)
	active_np_ = active_np.reshape(NUM_PIX,-1,1)

	L_np_ = np.sum(L_np_*active_np_, axis=1).reshape(H,W,3)	
	return L_np_


def render_rays(integrator: mi.SamplingIntegrator,
			scene: mi.Scene,
			rays: mi.Ray3f,
			sensor: Union[int, mi.Sensor] = 0,
			seed: int = 0,
			spp: int = 0,
			develop: bool = True,
			evaluate: bool = True) -> mi.TensorXf:

	if not develop:
		raise Exception("develop=True must be specified when "
						"invoking AD integrators")

	if isinstance(sensor, int):
		sensor = scene.sensors()[sensor]

	# Disable derivatives in all of the following
	with dr.suspend_grad():
		# Prepare the film and sample generator for rendering
		sampler, spp = integrator.prepare(
			sensor=sensor,
			seed=seed,
			spp=spp,
			aovs=integrator.aovs()
		)

		# Generate a set of rays starting at the sensor
		_, weight, pos, _ = integrator.sample_rays(scene, sensor, sampler)

		# Launch the Monte Carlo sampling process in primal mode
		L, valid, state = integrator.sample(
			mode=dr.ADMode.Primal,
			scene=scene,
			sampler=sampler,
			ray=rays,
			depth=mi.UInt32(0),
			δL=None,
			state_in=None,
			reparam=None,
			active=mi.Bool(True)
		)

		# Prepare an ImageBlock as specified by the film
		block = sensor.film().create_block()

		# Only use the coalescing feature when rendering enough samples
		block.set_coalesce(block.coalesce() and spp >= 4)

		# Accumulate into the image block
		alpha = dr.select(valid, mi.Float(1), mi.Float(0))
		if mi.has_flag(sensor.film().flags(), mi.FilmFlags.Special):
			aovs = sensor.film().prepare_sample(L * weight, rays.wavelengths,
												block.channel_count(), alpha=alpha)
			block.put(pos, aovs)
			del aovs
		else:
			block.put(pos, rays.wavelengths, L * weight*alpha, alpha)

		# Explicitly delete any remaining unused variables
		del sampler, rays, weight, pos, L, valid, alpha
		gc.collect()

		# Perform the weight division and return an image tensor
		sensor.film().put_block(block)
		integrator.primal_image = sensor.film().develop()

		return integrator.primal_image


def visualize_integrator(L:mitsuba.cuda_ad_rgb.Color3f, active:drjit.cuda.ad.Bool, cam_params:list):
	""" visualize the radiance values generated by integrator"""
	plt.figure()
	img = integrator_to_np(L, active, cam_params)
	plt.imshow(img)