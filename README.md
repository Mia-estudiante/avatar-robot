
## Environment Used
* conda environment for each branch

[SAM 2: Segment Anything in Images and Videos](https://github.com/facebookresearch/sam2)  
[SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory](https://github.com/yangchris11/samurai/tree/master)  
[Language Segment-Anything](https://github.com/luca-medeiros/lang-segment-anything)  
[BLIP CAM:Self Hosted Live Image Captioning with Real-Time Video Stream](https://github.com/zawawiAI/BLIP_CAM)

## Set up
1. Clone the repository:
```
git clone -b florence https://github.com/Mia-estudiante/avatar-robot.git
cd avatar-robot
```
2. Install dependencies:
```
conda create -n [env_name]
pip install transformers einops timm
```
3. Run the application:
```
python captioning_florence.py
```
