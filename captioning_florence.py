import cv2
import torch
import argparse
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM

# model and processor initiation
model_id = "microsoft/Florence-2-base"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()

if torch.cuda.is_available():
    model.to("cuda")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task_prompt",
    type=str,
    default="<DETAILED_CAPTION>",
    help="(<CAPTION> / <DETAILED_CAPTION> / <MORE_DETAILED_CAPTION>)"
)
args = parser.parse_args()


def run_example(image, task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer

def wrap_text(text, font, scale, thickness, max_width, margin=10):
    words = text.split(' ')
    lines = []
    current = ""
    for word in words:
        temp = f"{current} {word}".strip()
        (width, _), _ = cv2.getTextSize(temp, font, scale, thickness)
        if width + margin*2 <= max_width:
            current = temp
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


frame_id = 0
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame")
            break
        
        if frame_id % 10 == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img_rgb)

            results = run_example(image, args.task_prompt)
            caption = results.get(args.task_prompt)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        max_w = frame.shape[1]
        lines = []
        # for raw_line in caption.split("\n"):
        #     wrapped = wrap_text(raw_line, font, scale, thickness, max_w)
        #     lines.extend(wrapped)
        lines = wrap_text(caption, font, scale, thickness, max_w)

        y0, dy = 20, 20
        for i, line in enumerate(lines):
            y = y0 + i * dy
            cv2.putText(frame, line, (10, y), font, scale, (255,255,255), thickness, cv2.LINE_8)
            
        frame_id += 1
        cv2.imshow('Caption using Florence', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()