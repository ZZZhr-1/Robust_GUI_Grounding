# python Uground/check_attack.py \
#     --model_path osunlp/UGround-V1-7B \
#     --screenspot_imgs output \
#     --screenspot_test datas \
#     --task all \
#     --max_pixels 1024

CUDA_VISIBLE_DEVICES=0 python scripts/check_attack.py \
    --model_path /data/home/zhr/models/UGround-V1-7B \
    --screenspot_imgs ../outputs/ug/untarget \
    --screenspot_test ../datas \
    --task web \
    --max_pixels 1024