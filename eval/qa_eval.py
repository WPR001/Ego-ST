import os
import re
import json
import ast
import argparse
from datetime import datetime
from tqdm import tqdm  
import openai




def safe_parse_evaluation(response_str):
    try:
        match = re.search(r'\{.*\}', response_str, re.DOTALL)
        if match:
            dict_str = match.group(0)
            evaluation = ast.literal_eval(dict_str)
            return evaluation
        else:
            return {}
    except Exception as e:
        print(f"Error in safe_parse_evaluation: {e}")
        return {}

def evaluate_sample(sample, api_key):
    forward_question = sample.get('question', '')
    forward_answer   = sample.get('forward_answer', '')
    forward_predict  = sample.get('predict', '')

    # reverse_question = sample.get('question', '')
    # reverse_answer   = sample.get('reverse_answer', '')
    # reverse_predict  = sample.get('predict', '')

    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intelligent chatbot specialized in evaluating egocentric video route description tasks. "
                        "Your task is to assess the prediction by dividing the evaluation into three parts: \n\n"
                        "1. Direction: Evaluate whether there is any direction change in the prediction that does not appear in the correct answer. "
                        "   If the prediction shows a direction change that is not mentioned in the correct answer, this part should be scored lower. \n\n"
                        "2. Landmark: Evaluate the landmarks mentioned. Do not penalize if the prediction mentions landmarks that are not in the correct answer; "
                        "   however, if a landmark that appears in the correct answer is missing in the prediction, the score for this part should be reduced. \n\n"
                        "3. Semantic: Evaluate the overall logical consistency and semantic similarity of the sentence. \n\n"
                        "After evaluating the three parts, calculate a total score. The final score for each sample is computed as the sum of the three part scores divided by 3 (i.e. the average score). "
                        "For each part, assign an integer score between 0 and 5."
                    )
                },
                # {
                #     "role": "user",
                #     "content": (
                #         "Please evaluate the following video navigation task (Reverse: from point B to point A):\n\n"
                #         f"Reverse Question: {reverse_question}\n"
                #         f"Reverse Correct Answer: {reverse_answer}\n"
                #         f"Reverse Predicted Answer: {reverse_predict}\n\n"
                #         "Provide your evaluation strictly as a Python dictionary string with the following keys:\n"
                #         "  'direction': integer score (0-5),\n"
                #         "  'landmark': integer score (0-5),\n"
                #         "  'semantic and logic': integer score (0-5).\n"
                #         "Do not include any additional text or explanation."
                #     )
                # }

                {
                    "role": "user",
                    "content": (
                        "Please evaluate the following video navigation task (Forward: from point A to point B):\n\n"
                        f"Forward Question: {forward_question}\n"
                        f"Forward Correct Answer: {forward_answer}\n"
                        f"Forward Predicted Answer: {forward_predict}\n\n"
                        "Provide your evaluation strictly as a Python dictionary string with the following keys:\n"
                        "  'direction': integer score (0-5),\n"
                        "  'landmark': integer score (0-5),\n"
                        "  'semantic and logic': integer score (0-5).\n"
                        "Do not include any additional text or explanation."
                    )
                }




            ]
        )
        response_str = response["choices"][0]["message"]["content"]
        evaluation = safe_parse_evaluation(response_str)
    except Exception as e:
        print(f"Error evaluating sample {sample.get('id', '')}: {e}")
        evaluation = {"direction": 0, "landmark": 0, "semantic": 0}
    return evaluation

def main():
    parser = argparse.ArgumentParser(
        description="Read a JSON file, evaluate all samples using the GPT API, and compute the average total score"
    )
    parser.add_argument("--input_json", required=True, help="Path to the input JSON file")
    parser.add_argument("--output_json", required=True, help="Path to the output JSON file for evaluation results")
    parser.add_argument("--api_key", required=True, help="OpenAI API key")
    args = parser.parse_args()
    

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = {}
    total_score = 0.0
    total_samples = 0


    for key, sample in tqdm(data.items(), total=len(data), desc="Evaluating samples"):

        if not sample.get("forward_answer", "").strip() or not sample.get("predict", "").strip():
            print(f"Skipping sample {key} due to empty forward_answer or predict")
            continue

        evaluation = evaluate_sample(sample, args.api_key)
        

        sample_total = (evaluation.get("direction", 0) +
                        evaluation.get("landmark", 0) +
                        evaluation.get("semantic and logic", 0)) / 3.0
        evaluation["total"] = sample_total
        
        results[key] = evaluation
        total_score += sample_total
        total_samples += 1
        

        current_avg = total_score / total_samples
        print(f"Sample {key} Evaluation: {evaluation}")
        print(f"Cumulative average total score: {current_avg:.2f} over {total_samples} samples")
    

    output_data = {
        "evaluations": results,
        "average_total_score": total_score / total_samples if total_samples > 0 else 0
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()


