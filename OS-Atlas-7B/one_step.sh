export CUDA_VISIBLE_DEVICES=0

python scripts/target.py \
    --model_path OS-Copilot/OS-Atlas-Base-7B \
    --screenspot_imgs ../datas/screenspotv2_image \
    --screenspot_test ../datas/screenspot_mobile_os_target.json \
    --output_path ../outputs/os/target\
    --max_pixels 1024

python scripts/target.py \
    --model_path OS-Copilot/OS-Atlas-Base-7B \
    --screenspot_imgs ../datas/screenspotv2_image \
    --screenspot_test ../datas/screenspot_desktop_os_target.json \
    --output_path ../outputs/os/target\
    --max_pixels 1024

python scripts/target.py \
    --model_path OS-Copilot/OS-Atlas-Base-7B \
    --screenspot_imgs ../datas/screenspotv2_image \
    --screenspot_test ../datas/screenspot_web_os_target.json \
    --output_path ../outputs/os/target\
    --max_pixels 1024

python scripts/untarget.py \
    --model_path OS-Copilot/OS-Atlas-Base-7B \
    --screenspot_imgs ../datas/screenspotv2_image \
    --screenspot_test ../datas/screenspot_mobile_os_target.json \
    --output_path ../outputs/os/untarget\
    --max_pixels 1024

python scripts/untarget.py \
    --model_path OS-Copilot/OS-Atlas-Base-7B \
    --screenspot_imgs ../datas/screenspotv2_image \
    --screenspot_test ../datas/screenspot_desktop_os_target.json \
    --output_path ../outputs/os/untarget\
    --max_pixels 1024

python scripts/untarget.py \
    --model_path OS-Copilot/OS-Atlas-Base-7B \
    --screenspot_imgs ../datas/screenspotv2_image \
    --screenspot_test ../datas/screenspot_web_os_target.json \
    --output_path ../outputs/os/untarget\
    --max_pixels 1024

python scripts/check_attack.py \
    --model_path OS-Copilot/OS-Atlas-Base-7B \
    --screenspot_imgs ../outputs/os/target \
    --screenspot_test ../datas \
    --task all \
    --max_pixels 1024

python scripts/check_attack.py \
    --model_path OS-Copilot/OS-Atlas-Base-7B \
    --screenspot_imgs ../outputs/os/untarget \
    --screenspot_test ../datas \
    --task all \
    --max_pixels 1024