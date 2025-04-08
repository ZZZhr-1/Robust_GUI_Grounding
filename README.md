## Environment

python 3.10

transformers==4.45.1

`pip install -r requirements.txt`

## Download ScreenSpot-v2

`bash download.sh` Download screenspot-v2 images to datas

Download Model `osunlp/UGround-V1-7B`&`OS-Copilot/OS-Atlas-Base-7B`

## Uground

replace `generation_config.json` & `preprocessor_config.json` for Uground

(Default huggingface model path: ~/.cache/huggingface/modules/transformers_modules/UGround-V1-7B)

cd Uground

### target

bash target.sh

```
CUDA_VISIBLE_DEVICES=3 python scripts/target.py \
     --model_path osunlp/UGround-V1-7B \
     --screenspot_imgs ../datas/screenspotv2_image \
     --screenspot_test ../datas/screenspotv2_web_ug_target.json \
     --output_path ../outputs/ug/target\
     --max_pixels 1024
```

model_path: path to model

screenspot_imgs: image path

screenspot_test: preprocessed json data path (mobile\desktop\web, ug\os)

output_path: image output path

max_pixels: max_piexl

### untarget

bash untarget.sh

```
CUDA_VISIBLE_DEVICES=3 python scripts/untarget.py \
     --model_path osunlp/UGround-V1-7B \
     --screenspot_imgs ../datas/screenspotv2_image \
     --screenspot_test ../datas/screenspotv2_web_ug_target.json \
     --output_path ../outputs/ug/target\
     --max_pixels 1024
```

## OS-Atlas-7B

cd OS-Atlas-7B

### target

bash target.sh

```
CUDA_VISIBLE_DEVICES=3 python scripts/target.py \
     --model_path OS-Copilot/OS-Atlas-Base-7B \
     --screenspot_imgs ../datas/screenspotv2_image \
     --screenspot_test ../datas/screenspot_web_os_target.json \
     --output_path ../outputs/os/target\
     --max_pixels 1024
```

### untarget

bash untarget.sh

```
CUDA_VISIBLE_DEVICES=3 python scripts/untarget.py \
     --model_path OS-Copilot/OS-Atlas-Base-7B \
     --screenspot_imgs ../datas/screenspotv2_image \
     --screenspot_test ../datas/screenspot_web_os_target.json \
     --output_path ../outputs/os/untarget\
     --max_pixels 1024
```

## Evaluate

Example

bash check.sh

```
CUDA_VISIBLE_DEVICES=3 python scripts/check_attack.py \
     --model_path OS-Copilot/OS-Atlas-Base-7B \
     --screenspot_imgs ../outputs/os/target \
     --screenspot_test ../datas \
     --task all \
     --max_pixels 1024
```

output: os_check_result.txt

## Infer

Example

bash infer.sh

```
CUDA_VISIBLE_DEVICES=3 python scripts/infer_screenspot.py \
    --model_path OS-Copilot/OS-Atlas-Base-7B \
    --screenspot_imgs ../datas/screenspotv2_image \
    --screenspot_test ../datas \
    --task all \
    --max_pixels 1024
```

output: os_infer_result.txt

## Citation

If you find our work helpful, please consider citing our paper:

```
@article{zhao2025robustnessgui,
      title={On the Robustness of GUI Grounding Models Against Image Attacks}, 
      author={Haoren Zhao and Tianyi Chen and Zhen Wang},
      year={2025},
      eprint={2504.04716},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2504.04716}, 
}
```
If you have any questions, please feel free to leave issues.