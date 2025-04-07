import json
import argparse
import os
import logging
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
from transformers.generation import GenerationConfig
from preprocess.point_extract import extract_bbox, pred_2_point

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--screenspot_imgs', type=str, required=True)
parser.add_argument('--screenspot_test', type=str, required=True)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--max_pixels', type=int, required=True)
args = parser.parse_args()

model = Qwen2VLForConditionalGeneration.from_pretrained(
    args.model_path, torch_dtype="auto", device_map="cuda:0"
).eval()
processor = AutoProcessor.from_pretrained(args.model_path)
model.generation_config = GenerationConfig.from_pretrained(args.model_path, trust_remote_code=True)
print("Load Success")
min_pixels=4*28*28
max_pixels=args.max_pixels*28*28


if args.task == "all":
    tasks = ["mobile", "desktop", "web"]
else:
    tasks = [args.task]
tasks_result = []
result = []
max_num_patchs = 0
for task in tasks:
    dataset = "screenspot_" + task + "_v2.json"
    screenspot_data = json.load(open(os.path.join(args.screenspot_test, dataset), 'r'))
    print("Num of sample: " + str(len(screenspot_data)))

    num_action = 0
    corr_action = 0
    text_correct = []
    icon_correct = []
    num_wrong_format = 0
    for j, item in tqdm(enumerate(screenspot_data)):
        num_action += 1
        filename = item["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)
        image = Image.open(img_path)
        if not os.path.exists(img_path):
            print("img not found")
            input()
            continue
        instruction = item["instruction"]
        bbox = item["bbox"]
        img_size = image.size
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path, "min_pixels": min_pixels, "max_pixels": max_pixels},
                    {"type": "text", "text": f"""
Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.
Description: {instruction}
Answer:"""},
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        # print(inputs.pixel_values.shape)
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        try:
            if 'box' in output_text:
                pred_bbox = extract_bbox(output_text)
                click_point = [(pred_bbox[0][0] + pred_bbox[1][0]) / 2, (pred_bbox[0][1] + pred_bbox[1][1]) / 2]
                click_point = [item / 1000 for item in click_point]
            else:
                click_point = pred_2_point(output_text)
                click_point = [item / 1000 for item in click_point]
            if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
                corr_action += 1
                if item["data_type"] == 'text':
                    text_correct.append(1)
                else:
                    icon_correct.append(1)
                logging.info("match " + str(corr_action / num_action))
            else:
                if item["data_type"] == 'text':
                    text_correct.append(0)
                else:
                    icon_correct.append(0)
                with open("temp2.txt", 'a+') as f:
                    f.write(str(filename)+' '+str(click_point)+'\n')
                logging.info("unmatch " + str(corr_action / num_action))
            result.append({"img_path": img_path, "text": instruction, "bbox": bbox, "pred": click_point,
                           "type": item["data_type"], "source": item["data_source"]})
        except:
            num_wrong_format += 1
            if item["data_type"] == 'text':
                text_correct.append(0)
            else:
                icon_correct.append(0)
            logging.info("Step: " + str(j) + " wrong format")

    logging.info("Action Acc: " + str(corr_action / num_action))
    logging.info("Total num: " + str(num_action))
    logging.info("Wrong format num: " + str(num_wrong_format))
    logging.info("Text Acc: " + str(sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0))
    logging.info("Icon Acc: " + str(sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0))

    text_acc = sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0
    icon_acc = sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0
    tasks_result.append([text_acc, icon_acc])

logging.info(tasks_result)
with open("Uground_infer_result.txt", 'a+') as f:
    f.write(str(tasks_result)+'\n')
