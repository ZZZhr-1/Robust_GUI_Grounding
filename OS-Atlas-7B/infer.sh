# python scripts/infer_screenspot.py \
#     --model_path OS-Copilot/OS-Atlas-Base-7B \
#     --screenspot_imgs ../datas/screenspotv2_image \
#     --screenspot_test ../datas \
#     --task all \
#     --max_pixels 1024

# CUDA_VISIBLE_DEVICES=3 python scripts/infer_screenspot.py \
#     --model_path /data/home/zhr/models/OS-Atlas-Base-7B \
#     --screenspot_imgs ../datas/screenspotv2_image \
#     --screenspot_test ../datas \
#     --task all \
#     --max_pixels 400
# [[0.896551724137931, 0.5971563981042654], [0.6701030927835051, 0.4], [0.717948717948718, 0.5270935960591133]]

CUDA_VISIBLE_DEVICES=3 python scripts/infer_screenspot.py \
    --model_path /data/home/zhr/models/OS-Atlas-Base-7B \
    --screenspot_imgs /data/home/zhr/Robust_GUI_Grounding/datas/noisy_images/color_jitter \
    --screenspot_test ../datas \
    --task all \
    --max_pixels 1024

CUDA_VISIBLE_DEVICES=3 python scripts/infer_screenspot.py \
    --model_path /data/home/zhr/models/OS-Atlas-Base-7B \
    --screenspot_imgs /data/home/zhr/Robust_GUI_Grounding/datas/noisy_images/contrast_adjusted \
    --screenspot_test ../datas \
    --task all \
    --max_pixels 1024

CUDA_VISIBLE_DEVICES=3 python scripts/infer_screenspot.py \
    --model_path /data/home/zhr/models/OS-Atlas-Base-7B \
    --screenspot_imgs /data/home/zhr/Robust_GUI_Grounding/datas/noisy_images/gaussian_blur \
    --screenspot_test ../datas \
    --task all \
    --max_pixels 1024

CUDA_VISIBLE_DEVICES=3 python scripts/infer_screenspot.py \
    --model_path /data/home/zhr/models/OS-Atlas-Base-7B \
    --screenspot_imgs /data/home/zhr/Robust_GUI_Grounding/datas/noisy_images/gaussian_noise \
    --screenspot_test ../datas \
    --task all \
    --max_pixels 1024

CUDA_VISIBLE_DEVICES=3 python scripts/infer_screenspot.py \
    --model_path /data/home/zhr/models/OS-Atlas-Base-7B \
    --screenspot_imgs /data/home/zhr/Robust_GUI_Grounding/datas/noisy_images/random_shift \
    --screenspot_test ../datas \
    --task all \
    --max_pixels 1024