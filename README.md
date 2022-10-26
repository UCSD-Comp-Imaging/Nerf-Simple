# Simplified NeRF Implementation

This is a simplified implementation of NeRF, without hierarchical sampling. The code is equipped with a tensorboard setup, and scripts to train a NeRF model and generate a set of novel views with the camera rotating on a spherical dome. 




## Setup 

```
git clone https://github.com/UCSD-Comp-Imaging/NeRF_CT
cd NeRF_CT 
pip install -r requirements.txt
```

Install training data for lego scene . Refer to [NeRF](https://www.matthewtancik.com/nerf) for other datasets. 
```
bash download_data.sh
```

## Training 
Modify the yaml file (for ex. lego.yaml) in `configs/` folder. The default lego yaml file fields are self explanatory. 

```
python3 train.py --config configs/lego.yaml
```
## Testing 
To render novel views, in the test section of the yaml file, set `animation=True` to render the video as shown in the example. 
```
python3 test.py --config configs/lego.yaml
```
## Roadmap 
[ ] Hierarchical Sampling 
[ ] Support for LLF data
[ ] Integrate ColMap for training on real datasets 
[ ] Versioning the packages in requirements