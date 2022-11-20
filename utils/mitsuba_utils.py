import mitsuba as mi 
mi.set_variant('cuda_ad_rgb')

from mitsuba import ScalarTransform4f as T
import numpy as np 


def gen_sensor(fov:float, pose:T):
	"""
	fov (float): fov of camera in degrees
	pose (mitsuba ScalarTransform4f object): pose of camera / sensor
	"""
	# Apply two rotations to convert from spherical coordinates to world 3D coordinates.
	# origin = T.rotate([0, 0, 1], phi).rotate([1, 0, 0], theta) @ mi.ScalarPoint3f([0, 0, r])

	return mi.load_dict({
		'type': 'perspective',
		'fov': fov,
		'to_world': pose,
		# .look_at(
		# 	origin=origin,
		# 	target=[0, 0, 0],
		# 	up=[0, 1, 0]
		# ),
		'sampler': {
			'type': 'independent',
			'sample_count': 256
		},
		'film': {
			'type': 'hdrfilm',
			'width': 800,
			'height': 800,
			'rfilter': {
				'type': 'tent',
			},
			'pixel_format': 'rgb',
		},
	})

