CUDA_VISIBLE_DEVICES=1 python scripts/patch_attack.py \
    --model_path /data/home/zhr/models/UGround-V1-7B \
    --screenspot_imgs ../datas/screenspotv2_image \
    --screenspot_test ../datas/screenspotv2_mobile_ug_target.json \
    --output_path ../outputs/ug/patch_tl/256 \
    --max_pixels 256

CUDA_VISIBLE_DEVICES=1 python scripts/patch_attack.py \
    --model_path /data/home/zhr/models/UGround-V1-7B \
    --screenspot_imgs ../datas/screenspotv2_image \
    --screenspot_test ../datas/screenspotv2_desktop_ug_target.json \
    --output_path ../outputs/ug/patch_tl/256 \
    --max_pixels 256

CUDA_VISIBLE_DEVICES=1 python scripts/patch_attack.py \
    --model_path /data/home/zhr/models/UGround-V1-7B \
    --screenspot_imgs ../datas/screenspotv2_image \
    --screenspot_test ../datas/screenspotv2_web_ug_target.json \
    --output_path ../outputs/ug/patch_tl/256 \
    --max_pixels 256

CUDA_VISIBLE_DEVICES=1 python scripts/check_attack.py \
    --model_path /data/home/zhr/models/UGround-V1-7B \
    --screenspot_imgs ../outputs/ug/patch_tl/256 \
    --screenspot_test ../datas \
    --task all \
    --max_pixels 256