import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

device="cuda" if torch.cuda.is_available() else "cpu"
class Blip:
    def build_model(self, ckpt_path = None, device=device):
        model_id = "Salesforce/blip-image-captioning-base" if ckpt_path is None else ckpt_path
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id).to(device)

    def generate(self, roi):

        inputs = self.processor(roi, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_length=20,
                num_beams=3
            )
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption