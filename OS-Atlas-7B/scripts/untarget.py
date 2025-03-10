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
# scaling_tensor = scaling_tensor.reshape((3, 1, 1)).unsqueeze(0).to('cuda')
scaling_tensor = scaling_tensor.reshape((3, 1, 1)).to('cuda')
iters = 100
step_size = 1
epsilon = 16
step_size = step_size / 255.0 / scaling_tensor
epsilon = epsilon / 255.0 / scaling_tensor
inverse_normalize = torchvision.transforms.Normalize(mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711], std=[1.0 / 0.26862954, 1.0 / 0.26130258, 1.0 / 0.27577711])
output_path = args.output_path
os.makedirs(output_path, exist_ok=True)
criterion = torch.nn.MSELoss()

for idx, batch in enumerate(train_dataloader):
    input_ids = batch['input_ids'].to(model.device)
    attention_mask = batch['attention_mask'].to(model.device)
    labels = batch['labels'].to(model.device)
    pixel_values = batch['pixel_values'].to(model.device)
    image_grid_thw = batch['image_grid_thw'].to(model.device)

    filenames = batch['filenames'][0]
    ids = batch['ids']

    print(filenames)
    print(ids)

    print(pixel_values.shape)
    grid_t, grid_h, grid_w = image_grid_thw[0][0], image_grid_thw[0][1], image_grid_thw[0][2]
    # print(grid_t, grid_h, grid_w)
    restored = restore_images(pixel_values, grid_t=grid_t, grid_h=grid_h, grid_w=grid_w, merge_size=2, temporal_patch_size=2, patch_size=14, channel=3, data_format='channels_first')[0]
    with torch.no_grad():
        org_embedding = model.visual(pixel_values, image_grid_thw)
        # print(org_embedding.shape)
    delta = torch.randn_like(restored, requires_grad=True).to(model.device)
    iter_bar = tqdm(range(iters), desc="Adversarial Iterations", leave=False)
    for i in iter_bar:
        adv_images = restored + delta
        # print(adv_images.dtype)
        # print(adv_images.shape, adv_images.requires_grad)
        pixels, thw = pixel_reshape(image=adv_images, patch_size=14, merge_size=2, temporal_patch_size=2)
        # print(pixels, thw)
        adv_embedding = model.visual(pixels, image_grid_thw)
        # print(adv_embedding)
        loss = criterion(org_embedding, adv_embedding)
        loss.backward()
        # print(delta.grad)
        delta_data = torch.clamp(delta + step_size * torch.sign(delta.grad.detach()), -epsilon, epsilon)
    #     print(delta_data.shape)
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
    # generated_ids = model.generate(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         pixel_values=adv_images,
    #         image_grid_thw=image_grid_thw
    #     )
    # generated_ids_trimmed = [
    #         out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
    # ]
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    # )[0]
    # print(output_text)
    # break