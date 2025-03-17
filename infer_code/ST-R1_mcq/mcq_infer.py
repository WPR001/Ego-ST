import json
import os
import re
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def extract_choice(response):
    response = response.strip()

    match = re.match(r'^([A-D])\b', response)
    if match:
        return match.group(1)

    for char in response:
        if char in 'ABCD':
            return char
    return response  




model_path = "/path/" # R1 model

GPU="cuda:0"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=GPU,
)
processor = AutoProcessor.from_pretrained(model_path)

input_json_files = [

    '/path/Part4_forward_mcq_action.json',
    # 如有更多 JSON 文件，继续添加
]
video_folders = [

    '/path/part4_clip',
]

combined_results = {}


for idx, (json_file, video_folder) in enumerate(zip(input_json_files, video_folders)):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    

    for key, sample in data.items():
        if "options" not in sample["forward_mcq"] or not sample["forward_mcq"]["options"]:
            print(f"Skipping sample {key} due to empty options.")
            sample["predict"] = "Skipped due to empty options."
            combined_results[f"{idx+1}_{key}"] = sample
            continue

        video_file = sample.get("video")
        if not video_file:
            sample["predict"] = "No video file specified."
            combined_results[f"{idx+1}_{key}"] = sample
            continue
        video_path = os.path.join(video_folder, video_file)
        

        if "forward_mcq" in sample and "question" in sample["forward_mcq"]:
            question_text = sample["forward_mcq"]["question"]
        elif "question" in sample:
            question_text = sample["question"]
        else:
            sample["predict"] = "No question provided."
            combined_results[f"{idx+1}_{key}"] = sample
            continue

        options = sample["forward_mcq"]["options"]
        options_str = "\n".join(options)

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
                    {"type": "text", "text": question_text + "Options:" + options_str + "\n\nInstructions: Please select the most appropriate answer from the options above and return only the option number (e.g. A, B, C or D)." \
                      + "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>."},
                ],
            }
        ]
        
        try:

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
            inputs = inputs.to(GPU)
            

            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = output_text[0] if output_text else ""
            
            match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if match:
                answer_content = match.group(1).strip()
            else:
                answer_content = response.strip()
            
            predicted_choice = extract_choice(answer_content)
            sample["predict"] = predicted_choice
            print(f"Processed sample {key}: Prediction: {predicted_choice}")
        except Exception as e:
            print(f"Error processing sample {key}: {e}")
            sample["predict"] = "Error during model inference."
        
        # 将当前样本保存到合并字典中（key 格式可根据需要修改）
        combined_results[f"{idx+1}_{key}"] = sample


output_path = '/path/forward_mcq_output_action.json'

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(combined_results, f, ensure_ascii=False, indent=4)

print("All samples processed and saved to", output_path)