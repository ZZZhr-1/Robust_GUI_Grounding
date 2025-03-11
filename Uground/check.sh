# python Uground/check_attack.py \
#     --model_path osunlp/UGround-V1-7B \
#     --screenspot_imgs output \
#     --screenspot_test datas \
#     --task all \
#     --max_pixels 1024

# CUDA_VISIBLE_DEVICES=0 python scripts/check_attack.py \
#     --model_path /data/home/zhr/models/UGround-V1-7B \
#     --screenspot_imgs ../outputs/ug/untarget \
#     --screenspot_test ../datas \
#     --task web \
#     --max_pixels 1024
# [[0.6495726495726496, 0.46798029556650245]] web untarget
# [[0.0, 0.0]]

# CUDA_VISIBLE_DEVICES=0 python scripts/check_attack.py \
#     --model_path /data/home/zhr/models/UGround-V1-7B \
#     --screenspot_imgs ../outputs/ug/untarget \
#     --screenspot_test ../datas \
#     --task desktop \
#     --max_pixels 1024
# [[0.5154639175257731, 0.2571428571428571]]
# [[0.0, 0.0]]

CUDA_VISIBLE_DEVICES=3 python scripts/check_attack.py \
    --model_path /data/home/zhr/models/UGround-V1-7B \
    --screenspot_imgs ../outputs/ug/untarget \
    --screenspot_test ../datas \
    --task mobile \
    --max_pixels 1024
# [[0.7448275862068966, 0.46919431279620855]]
# [[0.0, 0.0]]

# CUDA_VISIBLE_DEVICES=0 python scripts/check_attack.py \
#     --model_path /data/home/zhr/models/UGround-V1-7B \
#     --screenspot_imgs ../outputs/ug/target/256 \
#     --screenspot_test ../datas \
#     --task web \
#     --max_pixels 256
# [[0.2264957264957265, 0.1477832512315271]]
# [[0.20085470085470086, 0.30049261083743845]]


CUDA_VISIBLE_DEVICES=3 python scripts/check_attack.py \
    --model_path /data/home/zhr/models/UGround-V1-7B \
    --screenspot_imgs ../outputs/ug/target/256 \
    --screenspot_test ../datas \
    --task all \
    --max_pixels 256
# [[0.5896551724137931, 0.27488151658767773], [0.17010309278350516, 0.16428571428571428], [0.2264957264957265, 0.1477832512315271]]
# [[0.05517241379310345, 0.15165876777251186], [0.30412371134020616, 0.38571428571428573], [0.20085470085470086, 0.30049261083743845]]

CUDA_VISIBLE_DEVICES=3 python scripts/check_attack.py \
    --model_path /data/home/zhr/models/UGround-V1-7B \
    --screenspot_imgs ../outputs/ug/untarget/256 \
    --screenspot_test ../datas \
    --task all \
    --max_pixels 256
# [[0.41724137931034483, 0.2890995260663507], [0.2268041237113402, 0.09285714285714286], [0.10256410256410256, 0.15270935960591134]]
# [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

