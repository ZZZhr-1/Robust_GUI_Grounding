# CUDA_VISIBLE_DEVICES=3 python scripts/target.py \
#     --model_path osunlp/UGround-V1-7B \
#     --screenspot_imgs ../datas/screenspotv2_image \
#     --screenspot_test ../datas/screenspotv2_web_ug_target.jsonn \
#     --output_path ../outputs/ug/target\
#     --max_pixels 1024

# CUDA_VISIBLE_DEVICES=3 python scripts/target.py \
#     --model_path /data/home/zhr/models/UGround-V1-7B \
#     --screenspot_imgs ../datas/screenspotv2_image \
#     --screenspot_test ../datas/screenspotv2_web_ug_target.json \
#     --output_path ../outputs/ug/target/256\
#     --max_pixels 256
# [[0.2264957264957265, 0.1477832512315271]]
# [[0.20085470085470086, 0.30049261083743845]]

CUDA_VISIBLE_DEVICES=3 python scripts/target.py \
    --model_path /data/home/zhr/models/UGround-V1-7B \
    --screenspot_imgs ../datas/screenspotv2_image \
    --screenspot_test ../datas/screenspotv2_mobile_ug_target.json \
    --output_path ../outputs/ug/target/256\
    --max_pixels 256

CUDA_VISIBLE_DEVICES=3 python scripts/target.py \
    --model_path /data/home/zhr/models/UGround-V1-7B \
    --screenspot_imgs ../datas/screenspotv2_image \
    --screenspot_test ../datas/screenspotv2_desktop_ug_target.json \
    --output_path ../outputs/ug/target/256\
    --max_pixels 256

CUDA_VISIBLE_DEVICES=3 python scripts/untarget.py \
    --model_path /data/home/zhr/models/UGround-V1-7B \
    --screenspot_imgs ../datas/screenspotv2_image \
    --screenspot_test ../datas/screenspotv2_mobile_ug_target.json \
    --output_path ../outputs/ug/untarget/256\
    --max_pixels 256

CUDA_VISIBLE_DEVICES=3 python scripts/untarget.py \
    --model_path /data/home/zhr/models/UGround-V1-7B \
    --screenspot_imgs ../datas/screenspotv2_image \
    --screenspot_test ../datas/screenspotv2_desktop_ug_target.json \
    --output_path ../outputs/ug/untarget/256\
    --max_pixels 256

CUDA_VISIBLE_DEVICES=3 python scripts/untarget.py \
    --model_path /data/home/zhr/models/UGround-V1-7B \
    --screenspot_imgs ../datas/screenspotv2_image \
    --screenspot_test ../datas/screenspotv2_web_ug_target.json \
    --output_path ../outputs/ug/untarget/256\
    --max_pixels 256