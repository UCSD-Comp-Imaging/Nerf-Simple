import torch
def gamma(x, L=4):	
	vec = []
	for i in range(L):
		vec += [torch.sin(2**i * np.pi * x), torch.cos(2**i * np.pi * x)]
	vec = torch.stack(vec)
	return vec