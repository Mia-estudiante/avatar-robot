import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.utils import DEVICE

class Llama:
    def build_model(self, ckpt_path: str | None = None, device=DEVICE):
        model_id = "meta-llama/Llama-3.2-1B-Instruct" if ckpt_path is None else ckpt_path
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def get_response(self, query):
        prompt = "Extract the object from the query or determine the tool that meets the query's intent, ending with a period.\
            Example 1: Query: pick a tool for communicate. Object: iphone.\
            Example 2: Query: I am tired. Object: chair.\
            Provide the response as a word of the object's value as a string, with no additional text.\
            Example response: 'iphone.'"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": query}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(DEVICE)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
