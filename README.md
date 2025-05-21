## Introduction

[SAM 2: Segment Anything in Images and Videos](https://github.com/facebookresearch/sam2)  
[SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory](https://github.com/yangchris11/samurai/tree/master)  
[Language Segment-Anything](https://github.com/luca-medeiros/lang-segment-anything)  
[BLIP CAM:Self Hosted Live Image Captioning with Real-Time Video Stream](https://github.com/zawawiAI/BLIP_CAM)   
* input: task prompt(caption / detailed caption(default) / more detailed caption)
* output: captured frame with generated caption on the top   
## Environment Used
* conda environment for each branch


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
Default task prompt is set as "<DETAILED_CAPTION>"   
You can adjust the task prompt by giving it as an argument.
```
python captioning_florence.py --task_prompt "<CAPTION>"
python captioning_florence.py --task_prompt "<DETAILED_CAPTION>"
python captioning_florence.py --task_prompt "<MORE_DETAILED_CAPTION>"
```
