# CUDA_VISIBLE_DEVICES=3 python scripts/target.py \
#     --model_path osunlp/UGround-V1-7B \
#     --screenspot_imgs ../datas/screenspotv2_image \
#     --screenspot_test ../datas/screenspotv2_web_ug_target.jsonn \
#     --output_path ../outputs/ug/target\
#     --max_pixels 1024

CUDA_VISIBLE_DEVICES=3 python scripts/target.py \
    --model_path /data/home/zhr/models/UGround-V1-7B \
    --screenspot_imgs ../datas/screenspotv2_image \
    --screenspot_test ../datas/screenspotv2_web_ug_target.json \
    --output_path ../outputs/ug/target/256\
    --max_pixels 256