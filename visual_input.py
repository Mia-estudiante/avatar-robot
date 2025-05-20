import cv2
import numpy as np
import torch

from visualizer import Visualizer

from sam2.build_sam import build_sam2_object_tracker
from mouse import Selector
torch_dtype=torch.float16


# Set SAM2 Configuration
NUM_OBJECTS = 1
SAM_CHECKPOINT_FILEPATH = "./checkpoints/sam2.1_hiera_tiny.pt"
SAM_CONFIG_FILEPATH = "./configs/samurai/sam2.1_hiera_t.yaml"
DEVICE = 'cuda:0'

# Open Video Stream
video_stream = cv2.VideoCapture(0)

video_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))

# Initializing visualizer and models
visualizer = Visualizer(video_width=video_width, video_height=video_height)

sam = build_sam2_object_tracker(num_objects=NUM_OBJECTS,
                                config_file=SAM_CONFIG_FILEPATH,
                                ckpt_path=SAM_CHECKPOINT_FILEPATH,
                                device=DEVICE,
                                verbose=False
                                )

selected = False
selector = Selector()

with torch.inference_mode(), torch.autocast('cuda:0', dtype=torch.bfloat16):
    while video_stream.isOpened():

        ret, frame = video_stream.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not selected:
            selection = selector.select(frame)
            if selection["point"] is None and selection["box"] is None:
                continue
            if selection["point"] is not None:
                sam_out = sam.track_new_object(img=img, points=selection["point"])
            else:
                sam_out = sam.track_new_object(img=img, box=selection["box"])
            selected = True

            
        else:
            sam_out = sam.track_all_objects(img=img)
        
            ret, frame = video_stream.read()
            visualizer.add_frame(frame=frame, mask=sam_out['pred_masks'])
