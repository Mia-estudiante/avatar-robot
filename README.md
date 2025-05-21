## Introduction

[SAM 2: Segment Anything in Images and Videos](https://github.com/facebookresearch/sam2)  
[SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory](https://github.com/yangchris11/samurai/tree/master)  
[Language Segment-Anything](https://github.com/luca-medeiros/lang-segment-anything)  
[BLIP CAM:Self Hosted Live Image Captioning with Real-Time Video Stream](https://github.com/zawawiAI/BLIP_CAM)   
* SAM2; visual_input.py
  * input: point, bbox(you can select an object by clicking or bounding it)
  * output: mask overlayed frame
* SAM2; text_input.py
  * input: text
  * output: mask overlayed frame   

## Environment Used
* conda environment for each branch

## Set up
1. Clone the repository:
```
git clone -b sam2 https://github.com/Mia-estudiante/avatar-robot.git
cd avatar-robot
```
2. Install dependencies:
```
conda create -n [env_name] python=3.11
pip install -r requirements.txt
```
3. Run the application:
```
python visual_input.py
```
or
```
python text_input.py
```
