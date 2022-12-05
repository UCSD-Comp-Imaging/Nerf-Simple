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
from utils.phaseoptic import gen_phase_optic, phase_optic_rays
from utils.xyz import transform_rays, mitsuba_spherical_to_pose

def gen_sensor(r:float or list, phi: float, theta: float, fov:float, spp=256, width=800, height=800, scene_center=np.zeros(3)):
	"""
	r: if float r is radius, movement along z - 0,0,r. if list, r is a 3d point 
	fov (float): fov of camera in degrees
	pose (mitsuba ScalarTransform4f object): pose of camera / sensor
	source orign/target is hardcoded, beaware
	"""
	# Apply two rotations to convert from spherical coordinates to world 3D coordinates.
	if type(r) == list:
		origin = T.rotate([0, 1, 0], phi).rotate([1, 0, 0], -np.abs(theta))@mi.ScalarPoint3f(r)	
	else:
		origin = T.rotate([0, 1, 0], phi).rotate([1, 0, 0], -np.abs(theta))@mi.ScalarPoint3f([0, 0, r])
	# target = np.array([0.4, 0.45, 0.5]) - lego scene target 
	target = np.array(scene_center)
	origin = origin + target 
	cam_pose = T.translate(target).look_at(
			origin=origin,
			target=[0, 0, 0],
			up=[0, 1, 0]
		)
	pose_r = np.array(cam_pose.matrix)
	# pose_r1 = mitsuba_spherical_to_pose(r, theta, phi, scene_center)
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

def get_scene_center(scene):
	""" get center of all meshes in scene """
	center = np.zeros(3)
	for shape in scene.shapes()[1:]:
		minc = np.array(shape.bbox().min)
		maxc = np.array(shape.bbox().max)
		center += (minc + maxc)/2

	print(center / len(scene.shapes()))
	# return np.array([0.3 ,0.5 ,0.])
	return center / len(scene.shapes())


def uniform_sphere_sample(num_samples, zsign=1):
	""" sample num_samples on hemisphere (determined by zsign) of radius r,
	    return corresponding theta and phi values. """
	thetas = [] 
	phis = []
	while(len(thetas) < num_samples):
		v = np.zeros(3)
		while (np.linalg.norm(v) < 0.001 or np.sign(v[2]) != zsign):
			v = np.random.randn(3)
		v = v / np.linalg.norm(v)
		phi = 180*np.arctan2(v[1],v[0]) / np.pi
		theta = 180*np.arctan2(np.sqrt(v[0]**2 + v[1]**2), v[2]) / np.pi
		thetas.append(theta.astype(np.float32))
		phis.append(phi.astype(np.float32))
	return thetas, phis

def gen_sensors(r_range, phi_range, theta_range, fov, spp, width, height,
                random=True, scene_center=np.zeros(3)):
	"""
	fov (float) : field of view in degree 
	spp (int) : sample count for renderer
	width (int) : width of image to generate
	height (int) : height of image to generate
	phi_range (list) : [phi_min, phi_max, num_phi] to simulate num_phi values between phi_min and phi_max
	theta_range (list) : [theta_min, theta_max, num_theta] 
	r_range (list) : [r_min, r_max, num_r] to simulate
	random (bool): If true, sample sensor poses randomly, else uniformly in interval
	"""
	# r, theta, phi = 3, -40, 0
	num_samples = len(np.linspace(*phi_range))*len(np.linspace(*theta_range)) 
	print(f"num samples to generate: {num_samples}")
	if random:
		thetas, phis = uniform_sphere_sample(num_samples, zsign=1)
		# phis = np.random.uniform(*phi_range)
		rs = [r_range[0]]
		rs = np.random.uniform(*r_range)
		# thetas = np.random.uniform(*theta_range)
	else:
		phis = np.linspace(*phi_range)
		rs = np.linspace(*r_range)
		thetas = np.linspace(*theta_range)	
	sensor_dicts = []
	
	## TODO remove this if else using itertools permutation functions
	if not random:
		for r in rs:
			for theta in thetas:
				for phi in phis:
					sensor_dict = gen_sensor(r, phi, theta, fov, spp, width, height, scene_center)
					sensor_dicts.append(sensor_dict)
	else:
		for r in rs:
			for theta, phi in zip(thetas, phis):
				sensor_dict = gen_sensor(r, phi, theta, fov, spp, width, height, scene_center)
				sensor_dicts.append(sensor_dict)	

	## returing thetas and phis because converting back to r theta phi space is weird
	return sensor_dicts, [rs, thetas, phis]

def gen_mitsuba_data(scene_path, datapath, data_type, camera_angle, \
                     spp=32, width=800, height=800, r_range=[1.75,1.8, 1], \
					 phi_range=[30,360, 20], theta_range=[-30,-60, 1], random_poses=True):
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
	random_poses (bool): If true, sample sensor poses randomly everytime function is called
	"""
	dn_integrator = mi.load_dict({'type':'aov', 'aovs': 'dd.z:depth,nn:sh_normal'})
	train_mitsuba_json = {}
	train_mitsuba_json['camera_angle_x'] = camera_angle
	train_mitsuba_json['frames']= [] 
	savepath = os.path.join(datapath, data_type)
	if not os.path.exists(savepath):
		os.makedirs(savepath)

	scene = mi.load_file(scene_path)
	scene_center = get_scene_center(scene)
	train_mitsuba_json['scene_center'] = scene_center.tolist()
	train_mitsuba_json['zdir'] = 1
	fov = camera_angle*180/(np.pi)
	img = mi.render(scene)
	sensor_dicts, sph_params = gen_sensors(r_range, phi_range, theta_range, fov,
										   spp, width, height, random_poses, scene_center)
	
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
				"transform_matrix": sensor_pose,
				"r": sph_params[0][0].tolist(),
				"theta": sph_params[1][i].tolist(),
				"phi": sph_params[2][i].tolist()
			}
		)
	json.dump(train_mitsuba_json, open(os.path.join(datapath, f'transforms_{data_type}.json'),'w'))

def gen_mitsuba_phaseoptic_data(scene_path, datapath, data_type, \
								camera_angle, spp=32, width=800, \
								height=800, r_range=[1.75,1.8, 1], \
								phi_range=[30,360, 20], theta_range=[-30,-60, 1], \
								random_poses=True, phase_optic_params=None):
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
	scene_center = get_scene_center(scene)
	mitsuba_json['scene_center'] = scene_center.tolist()
	mitsuba_json['zdir'] = 1
	img = mi.render(scene)
	fov = camera_angle*180/(np.pi)
	f = width /(2 * np.tan(camera_angle/2))
	cam_params = [height, width, f]
	phase_optic_params['cam_params'] = cam_params
	phase_optic_params['zdir'] = 1 ## since using mitsuba to generate data
	phase_optic = gen_phase_optic(**phase_optic_params)
	rays_phase_optic = phase_optic_rays(cam_params, phase_optic, spp, zdir=1)
	## generating rays exiting the phase optic
	sensor_dicts, sph_params = gen_sensors(r_range, phi_range, theta_range, fov, spp,
										   width, height, random_poses, scene_center)
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
				"r": sph_params[0][0].tolist(),
				"theta": sph_params[1][i].tolist(),
				"phi": sph_params[2][i].tolist()
			}
		)
	mitsuba_json['phase_optic_centers'] = phase_optic.centers.tolist()
	mitsuba_json['phase_optic_radii'] = phase_optic.radii.tolist()
	mitsuba_json['phase_optic_params'] = phase_optic_params
	json.dump(mitsuba_json, open(os.path.join(datapath, f'transforms_{data_type}.json'),'w'))
	# return imgsf

## TODO Function copied from mitsuba codebase, modified slightly. Need to clean this up with doc strings, or maybe replace
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

### Kind of miscellaneous functions, keeping them temporarily
def integrator_to_np(L:mitsuba.cuda_ad_rgb.Color3f, active:drjit.cuda.ad.Bool, cam_params:list):
	H,W,f = cam_params
	NUM_PIX = H*W
	L_np = np.array(L)
	L_np_ = L_np.reshape(NUM_PIX,-1,3)

	active_np = np.array(active)
	active_np_ = active_np.reshape(NUM_PIX,-1,1)

	L_np_ = np.sum(L_np_*active_np_, axis=1).reshape(H,W,3)	
	return L_np_

def visualize_integrator(L:mitsuba.cuda_ad_rgb.Color3f, active:drjit.cuda.ad.Bool, cam_params:list):
	""" visualize the radiance values generated by integrator"""
	plt.figure()
	img = integrator_to_np(L, active, cam_params)
	plt.imshow(img)