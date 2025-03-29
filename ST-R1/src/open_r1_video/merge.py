from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from transformers import Qwen2VLForConditionalGeneration
import os

def merge_lora_to_base_model():
    model_name_or_path = '/path/Qwen2-VL-7B-Instruct'
    adapter_name_or_path = '/path/r1/ckpt/Qwen2-VL-7B-Video-GRPO/video-sft-cot-without12'
    save_path = '/path/r1/ckpt/Qwen2-VL-7B-Video-GRPO/merge_cot_mcq_without12'

    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_name_or_path,
        trust_remote_code=True,
        use_fast=False if config.model_type == 'llama' else True
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map="auto")
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)




if __name__ == '__main__':
    merge_lora_to_base_model()
