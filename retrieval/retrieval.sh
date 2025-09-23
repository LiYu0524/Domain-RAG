# CUDA_VISIBLE_DEVICES=1 nohup python clip100_resnet_style_all_shots.py --datasets ArTaxOr DIOR FISH NEU-DET UODD clipart1k NWPU_VHR_10 --shots 1 5 10 > log_retrival_cdfsod.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python clip100_resnet_style_all_shots.py --datasets NWPU_VHR_10 --shots 3 > log_retrival_vhr3.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python clip100_resnet_style_all_shots.py --datasets NWPU_VHR_10 --shots 5 > log_retrival_vhr5.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python clip100_resnet_style_all_shots.py --datasets NWPU_VHR_10 --shots 10 > log_retrival_vhr10.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python clip100_resnet_style_all_shots.py --datasets NWPU_VHR_10 --shots 20 > log_retrival_vhr20.log 2>&1 &
