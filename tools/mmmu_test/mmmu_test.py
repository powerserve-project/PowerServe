import json
import os
import random
import re
import time
from argparse import ArgumentParser

import numpy as np
import requests
from datasets import concatenate_datasets, load_dataset
from mmmu.mmmu.utils.data_utils import (CAT_SHORT2LONG, construct_prompt,
                                        load_yaml, save_json)
from PIL import PngImagePlugin
from tqdm import tqdm


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f" {choice}" in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def parse_img_path(text):
    matches = re.findall("<img='(.*?)'>", text)
    return matches


def process_single_sample(data):
    question = data["question"]
    o_imgs_paths = []
    for option in data["options"]:
        current_o_imgs_paths = parse_img_path(option)
        for img_path in current_o_imgs_paths:
            o_imgs_paths.append(img_path)
    images = [
        data["image_1"],
        data["image_2"],
        data["image_3"],
        data["image_4"],
        data["image_5"],
        data["image_6"],
        data["image_7"],
    ]
    return {
        "id": data["id"],
        "question": question,
        "options": data["options"],
        "answer": data["answer"],
        "image": images,
        "question_type": data["question_type"],
    }


def run_model(args, samples):
    header = {"Content-type": "application/json"}

    out_samples = dict()
    for i, sample in tqdm(enumerate(samples)):
        if i % 300 == 0 and i:
            time.sleep(300)
        images: PngImagePlugin.PngImageFile = sample["image"]
        for idx, image in enumerate(images):
            if image is None:
                break
            with open(f"tem{idx}.png", "wb") as f:
                image.convert("RGB").save(f)
            assert (
                os.system(
                    f"sshpass -f /qnn/p.txt scp -o StrictHostKeyChecking=no -P 8022 tem{idx}.png {args.device_url}:{args.device_root}/tem{idx}.png"
                )
                == 0
            )
            pattern = f"<image {idx+1}>"
            replacement = f"<img>{args.device_root}/tem{idx}.png</img>"
            final_input_prompt = re.sub(pattern, replacement, sample["final_input_prompt"])
        payload = {"prompt": final_input_prompt, "n_predict": 128, "temperature": 0, "repeat_penalty": 1.0}
        try:
            response = requests.post(
                url=f"http://{args.host}:{args.port}/completion", json=payload, headers=header, timeout=10
            )
            assert response.status_code == 200
            res = json.loads(response.text)
            if sample["question_type"] == "multiple-choice":
                pred_ans = parse_multi_choice_response(res["content"], sample["all_choices"], sample["index2ans"])
            else:  # open question
                pred_ans = res["content"].replace("[end of text]", "")
        except Exception as e:
            print(e, flush=True)
            pred_ans = ""
        out_samples[sample["id"]] = pred_ans
        save_json(args.output_path, out_samples)
    return out_samples


def main():
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=str, default="test.json", help="name of saved json")
    parser.add_argument("--config_path", type=str, default="mmmu/mmmu/configs/llava1.5.yaml")
    parser.add_argument("--data_path", type=str, default="MMMU/MMMU")  # hf dataset path.
    parser.add_argument("--data_cache_path", type=str, default="")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=str, default="10000")
    parser.add_argument("--device_root", type=str, default="")
    parser.add_argument("--device_url", type=str, default="")
    args = parser.parse_args()

    # load config and process to one value
    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != "eval_params" and type(value) == list:
            assert len(value) == 1, "key {} has more than one value".format(key)
            args.config[key] = value[0]

    # run for each subject
    sub_dataset_list = []
    for subject in CAT_SHORT2LONG.values():
        sub_dataset = load_dataset(args.data_path, subject, split=args.split, cache_dir=args.data_cache_path)
        sub_dataset_list.append(sub_dataset)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)

    samples = []
    for sample in dataset:
        sample = process_single_sample(sample)

        sample = construct_prompt(sample, args.config)
        samples.append(sample)

    # run ex
    out_samples = run_model(args, samples)

    save_json(args.output_path, out_samples)


if __name__ == "__main__":
    main()
    os.system(
        "python mmmu/mmmu/main_eval_only.py --answer_path ./mmmu/mmmu/answer_dict_val.json --output_path ./test.json"
    )
