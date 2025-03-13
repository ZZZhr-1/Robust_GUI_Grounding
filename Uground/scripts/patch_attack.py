from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from preprocess.data_process import make_supervised_data_module
from preprocess.params import DataArguments
from preprocess.patch_image import restore_images, pixel_reshape
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision
from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--screenspot_imgs', type=str, required=True)
parser.add_argument('--screenspot_test', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--max_pixels', type=int, required=True)
args = parser.parse_args()


model_id = args.model_path
# Default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, device_map="cuda:0", torch_dtype=torch.bfloat16
).eval()
for name, param in model.named_parameters():
    param.requires_grad = False
processor = AutoProcessor.from_pretrained(model_id)

data_args = DataArguments(
    data_path=args.screenspot_test,
    image_min_pixels=4 * 28 * 28,
    image_max_pixels=args.max_pixels * 28 * 28,
    # image_max_pixels=1024*1024,
    image_folder=args.screenspot_imgs
)

# print(processor.tokenizer.pad_token_id)
sft_dataset = make_supervised_data_module(model_id, processor, data_args)
dataset = sft_dataset['train_dataset']

data_collator = sft_dataset['data_collator']

train_dataloader = DataLoader(
    dataset, 
    batch_size=1,
    shuffle=False, 
    collate_fn=data_collator
)

tokenizer = processor.tokenizer
scaling_tensor = torch.tensor((0.26862954, 0.26130258, 0.27577711)).to('cuda')
scaling_tensor = scaling_tensor.reshape((3, 1, 1)).to('cuda')
iters = 100
step_size = 2.5
epsilon = 255
step_size = step_size / 255.0 / scaling_tensor
epsilon = epsilon / 255.0 / scaling_tensor
inverse_normalize = torchvision.transforms.Normalize(mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711], std=[1.0 / 0.26862954, 1.0 / 0.26130258, 1.0 / 0.27577711])
output_path = args.output_path
os.makedirs(output_path, exist_ok=True)

for idx, batch in enumerate(train_dataloader):
    input_ids = batch['input_ids'].to(model.device)
    attention_mask = batch['attention_mask'].to(model.device)
    labels = batch['labels'].to(model.device)
    pixel_values = batch['pixel_values'].to(model.device)
    image_grid_thw = batch['image_grid_thw'].to(model.device)

    filenames = batch['filenames'][0]
    ids = batch['ids']

    grid_t, grid_h, grid_w = image_grid_thw[0][0], image_grid_thw[0][1], image_grid_thw[0][2]
    restored = restore_images(pixel_values, grid_t=grid_t, grid_h=grid_h, grid_w=grid_w, merge_size=2, temporal_patch_size=2, patch_size=14, channel=3, data_format='channels_first')[0]
    delta = torch.zeros_like(restored, requires_grad=True).to(model.device)
    mask = torch.zeros_like(restored, requires_grad=False).to(model.device)
    _, H, W = restored.shape  # 获取高度和宽度
    mask_h = int(0.1 * H)
    mask_w = int(0.1 * W)

    mask[:, :mask_h, :mask_w] = 1

    iter_bar = tqdm(range(iters), desc="Adversarial Iterations")
    for i in iter_bar:
        adv_images = restored + delta
        pixels, thw = pixel_reshape(image=adv_images, patch_size=14, merge_size=2, temporal_patch_size=2)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixels,
            image_grid_thw=image_grid_thw
        )
        loss = outputs.loss
        loss.backward()
        delta_data = torch.clamp(delta - step_size * torch.sign(delta.grad.detach() * mask), -epsilon, epsilon)
        delta.data = delta_data
        delta.grad.zero_()  # reset the gradient
        iter_bar.set_postfix(batch=f"{idx} / {len(train_dataloader)}", loss=loss.item())

    adv_images = restored + delta
    adv_images = torch.clamp(inverse_normalize(adv_images), 0.0, 1.0)

    adv_images = adv_images.cpu().detach().numpy().transpose((1, 2, 0))
    adv_images = (adv_images * 255).astype('uint8')
    adv_images = Image.fromarray(adv_images)
    adv_images_file = ids[0] + '_' + filenames[0]
    save_path = os.path.join(output_path, adv_images_file)
    adv_images.save(save_path)
