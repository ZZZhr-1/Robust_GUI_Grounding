# python scripts/infer_screenspot.py \
#     --model_path osunlp/UGround-V1-7B \
#     --screenspot_imgs ../datas/screenspotv2_image \
#     --screenspot_test ../datas \
#     --task all \
#     --max_pixels 1024

# CUDA_VISIBLE_DEVICES=2 python scripts/infer_screenspot.py \
#     --model_path /data/home/zhr/models/UGround-V1-7B \
#     --screenspot_imgs ../datas/screenspotv2_image \
#     --screenspot_test ../datas \
#     --task all \
#     --max_pixels 256

# CUDA_VISIBLE_DEVICES=2 python scripts/infer_screenspot.py \
#     --model_path /data/home/zhr/models/UGround-V1-7B \
#     --screenspot_imgs ../datas/screenspotv2_image \
#     --screenspot_test ../datas \
#     --task all \
#     --max_pixels 784

# CUDA_VISIBLE_DEVICES=2 python scripts/infer_screenspot.py \
#     --model_path /data/home/zhr/models/UGround-V1-7B \
#     --screenspot_imgs ../datas/screenspotv2_image \
#     --screenspot_test ../datas \
#     --task all \
#     --max_pixels 576

# CUDA_VISIBLE_DEVICES=2 python scripts/infer_screenspot.py \
#     --model_path /data/home/zhr/models/UGround-V1-7B \
#     --screenspot_imgs ../datas/screenspotv2_image \
#     --screenspot_test ../datas \
#     --task all \
#     --max_pixels 400

CUDA_VISIBLE_DEVICES=3 python scripts/infer_screenspot.py \
    --model_path /data/home/zhr/models/UGround-V1-7B \
    --screenspot_imgs /data/home/zhr/Robust_GUI_Grounding/datas/noisy_images/color_jitter \
    --screenspot_test ../datas \
    --task all \
    --max_pixels 1024

CUDA_VISIBLE_DEVICES=3 python scripts/infer_screenspot.py \
    --model_path /data/home/zhr/models/UGround-V1-7B \
    --screenspot_imgs /data/home/zhr/Robust_GUI_Grounding/datas/noisy_images/contrast_adjusted \
    --screenspot_test ../datas \
    --task all \
    --max_pixels 1024

CUDA_VISIBLE_DEVICES=3 python scripts/infer_screenspot.py \
    --model_path /data/home/zhr/models/UGround-V1-7B \
    --screenspot_imgs /data/home/zhr/Robust_GUI_Grounding/datas/noisy_images/gaussian_blur \
    --screenspot_test ../datas \
    --task all \
    --max_pixels 1024

CUDA_VISIBLE_DEVICES=3 python scripts/infer_screenspot.py \
    --model_path /data/home/zhr/models/UGround-V1-7B \
    --screenspot_imgs /data/home/zhr/Robust_GUI_Grounding/datas/noisy_images/gaussian_noise \
    --screenspot_test ../datas \
    --task all \
    --max_pixels 1024

CUDA_VISIBLE_DEVICES=3 python scripts/infer_screenspot.py \
    --model_path /data/home/zhr/models/UGround-V1-7B \
    --screenspot_imgs /data/home/zhr/Robust_GUI_Grounding/datas/noisy_images/random_shift \
    --screenspot_test ../datas \
    --task all \
    --max_pixels 1024