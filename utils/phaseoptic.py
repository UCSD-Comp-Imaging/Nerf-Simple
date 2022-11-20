import visu3d as v3d
import numpy as np
import math 
from utils.xyz import rays_single_cam


def random_lenslet_params(num_lenslets: int, cam_params: list, rscale: float, r_range_percent: float, same_sag: bool):
	""" return centers and radii of microlenslet in front of a sensor with 
		random lens radii and lens centers. 
	Args:
		num_lenslets: Total number of lenslets on sensor plane (should be a square number)
		cam_params: [H,W,f]
		rscale: factor with which to scale radius
		r_range_percent: percentage value between 0 and 1. randomly samples radii between B(1 +- radius_range).
		B is base radius computed based on number of lenslets, and sensor size. 
		same_sag (bool): keep the sag on the lens array same for all lenslets, by adjusting lens centers 
	Returns:
		centers, radii 
	"""
	num = math.isqrt(num_lenslets)
	H,W,f = cam_params 
	h = H/f
	w = W/f
	base_radius = rscale*h / (2*num)
	rlow = base_radius*(1 - r_range_percent)
	rhigh = base_radius*(1 + r_range_percent)
	radius_range = np.random.uniform(rlow, rhigh, size=num_lenslets)
	ax, ay = np.meshgrid(np.linspace(-h/2,h/2,num), np.linspace(-w/2,w/2,num))
	zs = -1*np.ones_like(ax) # assumed that all lenslets are on the image plane 

	base_sag = base_radius
	if same_sag:
		zs += (radius_range - base_sag).reshape(ax.shape)

	centers = np.stack((ax,ay,zs)).T.reshape(-1,3)
	radii = radius_range
	return centers, radii

def unif_lenslet_params(num_lenslets: int, cam_params: list, rscale: float or np.array):
	""" returns centers and radii of a microlenslet in front of a sensor plate 
	Args:
		num_lenslets: Total number of lenslets on the sensor plane (should be a integer**2)
		cam_params: [H,W,f]
		rscale: factor with which to scale radius
	Returns
		centers: num_lenslets x 3
		radii: list of radii num_lenslets x 1 
	"""
	num = math.isqrt(num_lenslets)
	H,W,f = cam_params 
	h = H/f
	w = W/f
	ax, ay = np.meshgrid(np.linspace(-h/2,h/2,num), np.linspace(-w/2,w/2,num))
	zs = -1*np.ones_like(ax) # assumed that all lenslets are on the image plane 
	centers = np.stack((ax,ay,zs)).T.reshape(-1,3)
	radii = rscale*np.ones(num_lenslets)*h/(2*num)
	return centers, radii


def refract(rays, n, mu):
	""" returns refracted ray incident at surface with surface normal n and refractive index mu w.r.t air
	Args:
		rays: (nx3) Incident ray directions
		n: (nx3) normals at the points where rays hit surface 
		mu: (scalar) refractive index of air w.r.t surface
	Returns:
		t (nx3) : refracted rays
	"""
	i_norms = np.linalg.norm(rays, axis=1, keepdims=True)
	i = rays / i_norms
	n = n / np.linalg.norm(n,axis=1, keepdims=True) 
	ni = np.sum(n*i, axis=1,keepdims=True)
	t = mu*i + n*np.sqrt(1 - (mu**2)*(1 - ni**2)) - mu*n*ni
	t = t * i_norms
	return t 
	

def intersect_plane(rays, normal, point):
	""" computes intersection of rays with plane 
	Args:
		rays (Nx6 np.array): origin and direction of rays 
		normal (3, np.array): normal vector to the plane 
		point (3, np.array): a point that lies on the plane 
	Returns:
		Nx6 rays with new origin on plane and same direction as before
	""" 
	## TODO : Handle the edge case for ray parallel to plane
	ray_dirn = rays[:,3:]
	ray_origin = rays[:,:3]
	t = (normal.dot(point) - ray_origin.dot(normal))/ray_dirn.dot(normal)
	rays_out = np.concatenate((ray_origin + t.reshape(-1,1)*ray_dirn, ray_dirn),axis=1)
	return rays_out
	

def intersect_sphere(rays, center, radius):
	""" computes intersection of rays with sphere given by center and radius 
	Args:
		rays (Nx6 np.array): origin and direction of rays 
		center (3, np.array): center of sphere
		radius (scalar float): radius of sphere
	Returns:
		out (Nx6, Np.array): origin is same as input for rays that don't intersect
		valid: indices of rays which don't intersect with this sphere
	""" 
	rays_origin = rays[:,:3]
	rays_direction = rays[:,3:]
	rays_dirn = rays_direction / np.linalg.norm(rays_direction, axis=1, keepdims=True)
	b = 2 * np.sum(rays_dirn* (rays_origin - center.reshape(-1,3)), axis=1, keepdims=True)
	c = np.linalg.norm(rays_origin - center, axis=1, keepdims=True) ** 2 - radius ** 2
	delta = b ** 2 - 4 * c
	out = np.ones_like(rays_origin)
	t = (-b + np.sqrt(delta)) / 2
	v1 = delta > 0 
	v2 = t >=0
	valid = np.where(v1*v2 == True)[0]
	invalid = np.where(v1*v2 == False)[0]
	out[valid] = rays_origin[valid] + t[valid]*rays_dirn[valid]
	out[invalid] = rays_origin[invalid]
	rays_out = np.concatenate((out, rays_direction), axis=1)
	return rays_out, valid 


def intersect_sphere_batch(rays, centers, radii):
	""" compute intersection of rays with a batch of spheres 
	Args:
		rays (Nx6 np.array): origin and direction of rays 
		centers (Mx3, np.array): center of sphere
		radii (M, scalar float): radius of sphere
	Returns:
		out (Nx6, Np.array): origin is same as input for rays that don't intersect
		valid: indices of rays which don't intersect with this sphere
	""" 
	rays_origin = rays[:,:3]
	rays_direction = rays[:,3:]
	rays_dirn = rays_direction / np.linalg.norm(rays_direction, axis=1, keepdims=True)

	## reshaping for batching 
	N,_ = rays_origin.shape
	M = centers.shape[0]
	rays_origin = rays_origin.reshape(1,N,3)
	rays_dirn = rays_dirn.reshape(1,N,3)
	centers = centers.reshape(M,1,3)
	radii = radii.reshape(M,1,1)
	
	b = 2 * np.sum(rays_dirn * (rays_origin - centers), axis=2, keepdims=True)
	c = np.linalg.norm(rays_origin - centers, axis=2, keepdims=True) ** 2 - radii ** 2
	delta = b ** 2 - 4 * c

	rays_origin = rays_origin.squeeze(0)
	rays_dirn = rays_dirn.squeeze(0)

	out = np.ones_like(rays_origin)
	t = (-b + np.sqrt(delta)) / 2 # M,N,1
	v1 = delta > 0  # M,N,1
	v2 = t >=0 
	valid_int = v1*v2 
	t[~valid_int] = -np.Inf
	tmax = np.max(t, axis=0) # N,1 
	sphere_idx = np.argmax(t, axis=0).squeeze(-1) # N,1
	valid = np.where(tmax != -np.Inf)[0]
	invalid = np.where(tmax == -np.Inf)[0]

	out[valid] = rays_origin[valid] + tmax[valid]*rays_dirn[valid]
	out[invalid] = rays_origin[invalid]
	rays_out = np.concatenate((out, rays_direction), axis=1)
	return rays_out, valid, sphere_idx   


def visu3d_tracer(rays_list):
	""" creates a set of visual rays to be traced from list of Nx6 rays
	Args:
		rays_list (list [r0, r1, ... rn]): r0 - Nx6. 0:3 origin, 3: direction 
		r_i origin = end point of r_(i-1). Assumes r_i and r_(i-1) have diff origins 
	Returns rt - Nx6 : visu3d compatible rays for tracing 
	"""
	rt = []
	for idx, r in enumerate(rays_list[:-1]):
		o, d = r[:,:3] , r[:,3:]
		r1 = rays_list[idx+1]
		o1, d1 = r1[:,:3], r1[:,:3] 
		dist = np.linalg.norm(o1 - o, axis=1, keepdims=True)
		d = d * dist / np.linalg.norm(d,axis=1, keepdims=True)
		r[:,3:] = d
		rt.append(r)
	rt.append(rays_list[-1])
	rt = np.concatenate(rt,axis=0)
	poss = rt[:,:3]
	dirss = rt[:,3:]
	viz = v3d.Ray(pos=poss, dir=dirss)
	return viz # viz.fig to show visu3d plot 


class SinglePhaseOptic:
	def __init__(self, center, radius, n=1.5):
		self.center = center
		self.radius = radius 


class PhaseOptic:
	def __init__(self, centers, radii, mu=1.5):
		## TODO: Add parameter for plane position and point on plane 
		self.centers = centers # Mx3
		self.radii = radii # M, 
		self.mu = mu # refractive index
	def visualize(self):
		""" function to visualize profile of phase optic """
		pass

def raytrace_phaseoptic(cam_params, element, spp=1):
	## TODO convert this to torch code 
	""" Traces from camera origin to end of single layer phase optic 
	Args:
		cam_params: [H,W,f]
		element: instance of class PhaseOptic
		spp (samples per pixel) int: if more than 1, randomly sample n samples per pixel. 
	Returns:
		returns a dictionary with following keys: 
		'rays_trace':[r0, r2, r4] : 
			list of rays at every stage of propagation where refraction happens,
			starts r0 (rays shot from origin). r0, r2, r4 - Nx6
		'sphere_idxs': sphere_idx (N,)
			index of the sphere hit by each ray. To be used with valid intersects 
		'valid_intersects': valid (N,)
			 boolean for each ray, true when the ray hit a sphere/lens.	
	"""
	H,W,f = cam_params
	mu = element.mu ## mu is assumed to be w.r.t air 
	centers = element.centers # centers of spheres
	radii = element.radii 

	# Nx3 
	cam_rays = rays_single_cam([H,W,f], spp).T.numpy() 
	
	# Nx6 ,Nx:3 origin, Nx3: direction
	rays = np.concatenate((np.zeros_like(cam_rays), cam_rays),axis=1)  
	
	## r0 stores rays originating from camera. Image plane assumed at [0, 0, -1]
	r0 = rays
	rays = intersect_plane(rays, np.array([0., 0., -1.]), np.array([0., 0., -1.]))
	
	## r1 stores rays at the first intersection with imaging plane 
	r1 = rays
	plane_normals = np.zeros_like(cam_rays)
	## normal vector at each point on plane assumed to be [0, 0, -1]
	plane_normals[:,2] = -1
	rays_refrac = refract(rays[:,3:], plane_normals, 1/mu)
	rays = np.concatenate((rays[:,:3], rays_refrac), axis=1)
	
	## r2 stores rays after refraction at first incidence on back plane of phase optic 
	r2 = rays
	rays, valid, sphere_idx = intersect_sphere_batch(r2, centers, radii)
	
	## r3 stores rays at their intersection with the phaseoptic front surface (lens)
	r3 = rays

	## computing normals at the intersection points
	sphere_normals = rays[:, :3] - centers[sphere_idx]
	sphere_normals = sphere_normals / np.linalg.norm(sphere_normals,axis=1,keepdims=True)

	rays_refrac = refract(rays[valid][:, 3:], sphere_normals[valid], mu)
	rays[valid,3:] = rays_refrac
	
	## r4 stores final rays emitted by phase optic 
	r4 = rays
	out = {'rays_trace':[r0, r2, r4],
			'sphere_idxs':sphere_idx,
			'valid_intersects': valid }
	return out


def visu3d_tracer(rays_list):
	""" creates a set of visual rays to be traced from list of Nx6 rays
	Args:
		rays_list (list [r0, r1, ... rn]): r0 - Nx6. 0:3 origin, 3: direction 
		r_i origin = end point of r_(i-1). Assumes r_i and r_(i-1) have diff origins 
	Returns rt - Nx6 : visu3d compatible rays for tracing 
	"""
	rt = []
	for idx, r in enumerate(rays_list[:-1]):
		o, d = r[:,:3] , r[:,3:]
		r1 = rays_list[idx+1]
		o1, d1 = r1[:,:3], r1[:,:3] 
		dist = np.linalg.norm(o1 - o, axis=1, keepdims=True)
		d = d * dist / np.linalg.norm(d,axis=1, keepdims=True)
		r[:,3:] = d
		rt.append(r)
	rt.append(rays_list[-1])
	rt = np.concatenate(rt,axis=0)
	poss = rt[:,:3]
	dirss = rt[:,3:]
	viz = v3d.Ray(pos=poss, dir=dirss)
	return viz