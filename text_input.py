import cv2
import numpy as np
import torch

from PIL import Image
from visualizer import Visualizer
from models.gdino import GDINO
from models.llama import Llama

from sam2.build_sam import build_sam2_object_tracker
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

gdino = GDINO()
gdino.build_model()

llama = Llama()
llama.build_model()

def get_bbox(images_pil, texts_prompt, box_threshold, text_threshold):
    gdino_results = gdino.predict(images_pil, [texts_prompt], box_threshold, text_threshold)
    sam_boxes = []
    sam_indices = []
    for idx, result in enumerate(gdino_results):
        result = {k: (v.cpu().numpy() if hasattr(v, "numpy") else v) for k, v in result.items()}
        processed_result = {
            **result,
            "masks": [],
            "mask_scores": [],
        }

        sam_boxes.append(processed_result["boxes"])
        sam_indices.append(idx)

    return sam_boxes


first_frame = True
with torch.inference_mode(), torch.autocast('cuda:0', dtype=torch.bfloat16):
    while video_stream.isOpened():

        ret, frame = video_stream.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if first_frame:
            image = Image.fromarray(img)

            text_prompt = llama.get_response(input("What are you looking for? "))
            bbox = get_bbox([image], text_prompt, 0.3, 0.25)
            xyxy = bbox[0][0]
            bbox = [[xyxy[0], xyxy[1]], [xyxy[2], xyxy[3]]]
            bbox = np.array(bbox, dtype=np.float32)
            sam_out = sam.track_new_object(img=img, box=bbox)
            
            first_frame = False
            
        else:
            sam_out = sam.track_all_objects(img=img)
        
        ret, frame = video_stream.read()
        visualizer.add_frame(frame=frame, mask=sam_out['pred_masks'])
