## One-step

### Env

python 3.10

transformers==4.45.1

`pip install -r requirements.txt`

### pre

`bash download.sh` Download screenspot-v2 images to datas

Download Model `osunlp/UGround-V1-7B`&`OS-Copilot/OS-Atlas-Base-7B`

replace `generation_config.json` & `preprocessor_config.json` for Uground

(Default huggingface model path: ~/.cache/huggingface/modules/transformers_modules/UGround-V1-7B)

### run

need to run：

1. Uground target

```
cd Uground
bash one_step.sh
```

2. OS-Atlas-7B target & untarget

```
cd OS-Atlas-7B
bash one_step.sh
```

default cuda deivice `CUDA_VISIBLE_DEVICES=0`

default model path `OS-Copilot/OS-Atlas-Base-7B` & `osunlp/UGround-V1-7B`

(modify on one_step.sh)

### result

result on os_check_result.txt & Uground_check_result.txt

## Environment

python 3.10

transformers==4.45.1

`pip install -r requirements.txt`

## Download ScreenSpot-v2

`bash download.sh` Download screenspot-v2 images to datas

replace generation_config.json & preprocessor_config.json for Uground (Default huggingface model path: ~/.cache/huggingface/modules/transformers_modules/UGround-V1-7B)

## Uground

replace generation_config.json & preprocessor_config.json for Uground (Default huggingface model path: ~/.cache/huggingface/modules/transformers_modules/UGround-V1-7B)

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