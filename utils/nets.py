import numpy as np
import torch 
import torch.nn as nn
import warnings
from utils.xyz import *

class Nerf(nn.Module):
	def __init__(self, pos_enc=True, depth=8):
		super(Nerf, self).__init__()
		self.depth = depth 
		
	pass

class CoarseNet(nn.Module):
	pass

class FineNet(nn.Module):
	pass


