# python scripts/check_attack.py \
#     --model_path OS-Copilot/OS-Atlas-Base-7B \
#     --screenspot_imgs ../output/ \
#     --screenspot_test ../datas \
#     --task all \
#     --max_pixels 1024


# CUDA_VISIBLE_DEVICES=2 python scripts/check_attack.py \
#     --model_path /data/home/zhr/models/OS-Atlas-Base-7B \
#     --screenspot_imgs ../outputs/os/target/256 \
#     --screenspot_test ../datas \
#     --task web \
#     --max_pixels 256
# [[0.08974358974358974, 0.10344827586206896]]
# [[0.029914529914529916, 0.034482758620689655]]

# CUDA_VISIBLE_DEVICES=2 python scripts/check_attack.py \
#     --model_path /data/home/zhr/models/OS-Atlas-Base-7B \
#     --screenspot_imgs ../outputs/os/target/256 \
#     --screenspot_test ../datas \
#     --task all \
#     --max_pixels 256
# [[0.3724137931034483, 0.11848341232227488], [0.10824742268041238, 0.05], [0.08974358974358974, 0.10344827586206896]]
# [[0.020689655172413793, 0.04265402843601896], [0.020618556701030927, 0.07142857142857142], [0.029914529914529916, 0.034482758620689655]]

# CUDA_VISIBLE_DEVICES=2 python scripts/check_attack.py \
#     --model_path /data/home/zhr/models/OS-Atlas-Base-7B \
#     --screenspot_imgs ../outputs/os/untarget/256 \
#     --screenspot_test ../datas \
#     --task all \
#     --max_pixels 256
# [[0.4379310344827586, 0.16113744075829384], [0.15979381443298968, 0.1], [0.07692307692307693, 0.11822660098522167]]
# [[0.010344827586206896, 0.004739336492890996], [0.0, 0.0], [0.0, 0.0]]