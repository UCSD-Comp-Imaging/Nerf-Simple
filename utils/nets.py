import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import warnings
from utils.xyz import *

class Nerf(nn.Module):
	def __init__(self, Lp = 10, Ld = 4, H=256):
		super(Nerf, self).__init__()
		self.Ld = Ld 
		self.Lp = Lp
		in_Cx = Lp*6 +3
		in_Cd = Ld*6 +3
		layers = []
		layers = [nn.Linear(in_Cx, H), nn.ReLU()]
		for i in range(4):
			layers += [nn.Linear(H, H), nn.ReLU()]
		self.layers_0 = nn.Sequential(*layers)

		self.skip_conn_layer = nn.Sequential(nn.Linear(H+in_Cx, H), nn.ReLU())

		self.layers_1 = nn.Sequential(nn.Linear(H,H),
									  nn.ReLU(),
									  nn.Linear(H,H),
									  nn.ReLU())
		# self.sigma_fc = nn.Sequential(nn.Linear(H,1), nn.ReLU())
		self.sigma_fc = nn.Sequential(nn.Linear(H,1))
		self.layers_2 = nn.Linear(H,H)
		# self.color_fc = nn.Sequential(nn.Linear(H+in_Cd, H//2),
		#							  nn.ReLU(),
		#							  nn.Linear(H//2,3),
		#							  nn.Sigmoid())

		self.color_fc = nn.Sequential(nn.Linear(H+in_Cd, H//2),
									  nn.ReLU(),
									  nn.Linear(H//2,3))

	def forward(self, v):
		# v: B x 6
		x, d = positional_encoder(v, Lp=self.Lp, Ld=self.Ld)
		out = self.layers_0(x)
		out = self.skip_conn_layer(torch.cat([out, x],axis=1))
		out = self.layers_1(out)
		sigma = self.sigma_fc(out) ## should add zero mean and unit variance noise before passing to ReLU for slightly better perf 
		out = self.layers_2(out)
		color = self.color_fc(torch.cat((out, d),axis=1))
		return torch.cat((color, sigma),axis=1) 

class CoarseNet(nn.Module):
	pass

class FineNet(nn.Module):
	pass

class NerfRef(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=True):
        """ 
        """
        super(NerfRef, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))