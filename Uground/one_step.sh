export CUDA_VISIBLE_DEVICES=0

MODEL_PATH=osunlp/UGround-V1-7B
# MODEL_PATH=/data/home/zhr/models/UGround-V1-7B

python scripts/target.py \
    --model_path $MODEL_PATH \
    --screenspot_imgs ../datas/screenspotv2_image \
    --screenspot_test ../datas/screenspotv2_mobile_ug_target.json \
    --output_path ../outputs/ug/target/test\
    --max_pixels 1024

python scripts/target.py \
    --model_path $MODEL_PATH \
    --screenspot_imgs ../datas/screenspotv2_image \
    --screenspot_test ../datas/screenspotv2_desktop_ug_target.json \
    --output_path ../outputs/ug/target\
    --max_pixels 1024

python scripts/target.py \
    --model_path $MODEL_PATH \
    --screenspot_imgs ../datas/screenspotv2_image \
    --screenspot_test ../datas/screenspotv2_web_ug_target.json \
    --output_path ../outputs/ug/target\
    --max_pixels 1024

python scripts/check_attack.py \
    --model_path $MODEL_PATH \
    --screenspot_imgs ../outputs/ug/target \
    --screenspot_test ../datas \
    --task all \
    --max_pixels 1024

# python scripts/target.py \
#     --model_path osunlp/UGround-V1-7B \
#     --screenspot_imgs ../datas/screenspotv2_image \
#     --screenspot_test ../datas/screenspotv2_mobile_ug_target.json \
#     --output_path ../outputs/ug/target\
#     --max_pixels 1024

# python scripts/target.py \
#     --model_path osunlp/UGround-V1-7B \
#     --screenspot_imgs ../datas/screenspotv2_image \
#     --screenspot_test ../datas/screenspotv2_desktop_ug_target.json \
#     --output_path ../outputs/ug/target\
#     --max_pixels 1024

# python scripts/target.py \
#     --model_path osunlp/UGround-V1-7B \
#     --screenspot_imgs ../datas/screenspotv2_image \
#     --screenspot_test ../datas/screenspotv2_web_ug_target.json \
#     --output_path ../outputs/ug/target\
#     --max_pixels 1024

# python scripts/check_attack.py \
#     --model_path osunlp/UGround-V1-7B \
#     --screenspot_imgs ../outputs/ug/target \
#     --screenspot_test ../datas \
#     --task all \
#     --max_pixels 1024