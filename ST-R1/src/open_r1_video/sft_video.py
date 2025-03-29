# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os
import random
from dataclasses import dataclass, field
from typing import Any

import requests
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, Qwen2VLProcessor

from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config


def download_video(url: str, cache_dir: str) -> str:
    """Download video if not already present locally."""
    os.makedirs(cache_dir, exist_ok=True)  # Create cache dir if it doesn't exist
    filename = url.split("/")[-1]
    local_path = os.path.join(cache_dir, filename)

    if os.path.exists(local_path):
        return local_path

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return local_path
    except requests.RequestException as e:
        raise Exception(f"Failed to download video: {e}") from e


def prepare_dataset(example: dict[str, Any], cache_dir: str) -> dict[str, list[dict[str, Any]]]:
    """
    Prepare dataset example for training.
    """
    video_path = example["video"]
    question_text = example["original_question"]
    answer_text = example["solution"]
    # answer_text = example["original_answer"] # without cot


    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {
                    "type": "text",
                    "text": question_text,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": answer_text,
                }
            ],
        },
    ]
    return {"messages": messages}


def collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate batch of examples for training using the new input formatting."""
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for i, example in enumerate(examples):
        try:
            text = processor.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                example["messages"], return_video_kwargs=True
            )
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
            labels = inputs["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            visual_tokens = (
                [151652, 151653, 151656]
                if isinstance(processor, Qwen2VLProcessor)
                else [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
            )
            for vt in visual_tokens:
                labels[labels == vt] = -100

            input_ids_list.append(inputs["input_ids"])
            attention_mask_list.append(inputs["attention_mask"])
            labels_list.append(labels)
        except Exception as e:
            raise ValueError(f"Failed to process example {i}: {e}") from e


    batch = {
        "input_ids": torch.cat(input_ids_list, dim=0),
        "attention_mask": torch.cat(attention_mask_list, dim=0),
        "labels": torch.cat(labels_list, dim=0),
    }
    return batch


@dataclass
class CustomScriptArguments(ScriptArguments):
    r"""
    Arguments for the script.

    Args:
        video_cache_dir (`str`, *optional*, defaults to `"/tmp/videos/"`):
            Video cache directory.
    """
    video_cache_dir: str = field(default="/tmp/videos/", metadata={"help": "Video cache directory."})


if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((CustomScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Load dataset
    dataset = load_dataset("json", data_files=script_args.dataset_name, split="train")

    # Setup model
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    # Quantization configuration for 4-bit training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     bnb_8bit_quant_type="nf8",
    #     bnb_8bit_compute_dtype=torch.bfloat16,
    #     bnb_8bit_use_double_quant=True)



    # Model initialization
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        quantization_config=bnb_config,
    )

    model = AutoModelForVision2Seq.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # Configure model modules for gradients
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_reentrant = False
        model.enable_input_require_grads()

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    # Prepare dataset
    prepared_dataset = [prepare_dataset(example, script_args.video_cache_dir) for example in dataset]

    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.init(project="video-llm-training")

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        processing_class=processor,
    )

    # Train model
    trainer.train()

    # Save final model
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    wandb.finish()