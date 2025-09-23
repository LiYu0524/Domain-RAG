# CUDA_VISIBLE_DEVICES=5 nohup python lama_inpaint.py > log_inpaint.log 2>&1 &

# CUDA_VISIBLE_DEVICES=5 nohup python lama_inpaint.py --datasets clipart1k --shots 5 > log_inpaint_clipart1k_5shot.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python lama_inpaint.py --datasets NWPU_VHR-10 --shots 3 > log_inpaint_NWPU_VH_3shot.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python lama_inpaint.py --datasets NWPU_VHR-10 --shots 5 > log_inpaint_NWPU_VH_5shot.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python lama_inpaint.py --datasets NWPU_VHR-10 --shots 10 > log_inpaint_NWPU_VH_10shot.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python lama_inpaint.py --datasets NWPU_VHR-10 --shots 20 > log_inpaint_NWPU_VH_20shot.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python lama_inpaint.py --datasets Camouflage --shots 1 > log_inpaint_camouflage_1shot.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python lama_inpaint.py --datasets Camouflage --shots 2 > log_inpaint_camouflage_2shot.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python lama_inpaint.py --datasets Camouflage --shots 3 > log_inpaint_camouflage_3shot.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python lama_inpaint.py --datasets Camouflage --shots 5 > log_inpaint_camouflage_5shot.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python lama_inpaint.py --datasets coco --shots 5 > log_inpaint_coco_5shot.log 2>&1 &