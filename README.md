# Simplified NeRF Implementation

This is a simplified implementation of NeRF, without hierarchical sampling. The code is equipped with a tensorboard setup, and scripts to train a NeRF model and generate a set of novel views with the camera rotating on a spherical dome. 









https://user-images.githubusercontent.com/114626135/197914242-6efba591-2452-41ac-86bb-bd145aab65c2.mov


The video was generated by a model trained on 25 images, for 10000 iterations (takes around 30 min). 


This drive [link](https://drive.google.com/drive/folders/1upzp3VQQSBWM8U182LmzmxcpjQ-Ou4wb?usp=share_link) has all the other NeRF videos for the CSE 274 project, including the phase optic experiment videos.  

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
To start the tensorboard, open another terminal and run 
```
tensorboard --logdir=logs/
```
## Testing 
To render novel views, in the test section of the yaml file, set `animation=True` to render the video as shown in the example. 
```
python3 test.py --config configs/lego.yaml
```
## Roadmap 
- [ ] Hierarchical Sampling 
- [ ] Support for LLF data
- [ ] Integrate ColMap for training on real datasets 
- [ ] Versioning the packages in requirements





## Phase Optic results 

https://user-images.githubusercontent.com/114626135/205836930-aaa52649-9419-4827-b6c2-963bdce6ba90.mp4


https://user-images.githubusercontent.com/114626135/205836948-182975b9-e404-48a2-ac91-8adbfd63a5a9.mp4


https://user-images.githubusercontent.com/114626135/205836916-3bb19038-bfef-4089-8256-ee8b23231df8.mp4



