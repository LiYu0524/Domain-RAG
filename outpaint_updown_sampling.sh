process_id=1

CUDA_VISIBLE_DEVICES=0 nohup python outpainting_updown_sampling_redux.py --process_id ${process_id} --dataset coco  --shot 10 > coco_10shot_1gpu.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python outpainting_updown_sampling_redux.py --process_id ${process_id} --dataset clipart1k --shot 1 > ./log_outpaint/outpaint_updown_sampling_shot1_clipart1k_process${process_id}.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python outpainting_updown_sampling_redux.py --process_id ${process_id} --dataset UODD --shot 1 > ./log_outpaint/uodd_1shot_process${process_id}.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python outpainting_updown_sampling_redux.py --process_id ${process_id} --dataset FISH --shot 1 > ./log_outpaint/outpaint_updown_sampling_shot1_fish_process${process_id}_prompt.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python outpainting_updown_sampling_redux.py --process_id ${process_id} --dataset ArTaxOr --shot 1 > ./log_outpaint/outpaint_updown_sampling_shot1_artaxor_process${process_id}.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python outpainting_updown_sampling_redux.py --process_id ${process_id} --dataset DIOR --shot 1 > ./log_outpaint/outpaint_updown_sampling_shot1_dior_process${process_id}.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python outpainting_updown_sampling_redux.py --process_id ${process_id} --dataset NEU-DET --shot 1 > ./log_outpaint/outpaint_updown_sampling_shot1_neu-det_process${process_id}.log 2>&1 &


# CUDA_VISIBLE_DEVICES=2 nohup python outpainting_updown_sampling_redux.py --process_id ${process_id} --dataset_group 1 --shot 1 > outpaint_updown_sampling_1_${process_id}.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python outpainting_updown_sampling_redux.py --process_id ${process_id} --dataset_group 2 --shot 1 > outpaint_updown_sampling_2_${process_id}.log 2>&1 &


# CUDA_VISIBLE_DEVICES=5 nohup python outpainting_updown_sampling_redux.py --process_id 3 --dataset_group all --shot 1 > outpaint_updown_sampling_all.log 2>&1 &

# CUDA_VISIBLE_DEVICES=4 nohup python outpainting_updown_sampling_redux.py --process_id 1 --shot 1 > outpaint_updown_sampling_1.log 2>&1 &

# CUDA_VISIBLE_DEVICES=5 nohup python outpainting_updown_sampling_redux.py --process_id 2 --shot 1 > outpaint_updown_sampling_2.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python outpainting_updown_sampling_redux.py --process_id ${process_id} --dataset UODD --shot 1 > ./log_outpaint/uodd_1shot_process${process_id}.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python outpainting_updown_sampling_redux.py --process_id $process_id --dataset UODD --shot 5 > ./log_outpaint/uodd_5shot.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python outpainting_updown_sampling_redux.py --process_id $process_id --dataset UODD --shot 10 > ./log_outpaint/uodd_10shot.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python outpainting_updown_sampling_redux.py --process_id ${process_id} --dataset FISH --shot 1 > ./log_outpaint/outpaint_updown_sampling_shot1_fish_process${process_id}_prompt.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python outpainting_updown_sampling_redux.py --process_id $process_id --dataset FISH --shot 5 > outpaint_updown_sampling_shot5_fish_process$process_id.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python outpainting_updown_sampling_redux.py --process_id $process_id --dataset FISH --shot 10 > outpaint_updown_sampling_shot10_fish_process$process_id.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 nohup python outpainting_updown_sampling_redux.py --process_id ${process_id} --dataset ArTaxOr --shot 1 > ./log_outpaint/outpaint_updown_sampling_shot1_artaxor_process${process_id}.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python outpainting_updown_sampling_redux.py --process_id 4 --dataset ArTaxOr --shot 5 > outpaint_updown_sampling_shot5_artaxor.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python outpainting_updown_sampling_redux.py --process_id 4 --dataset ArTaxOr --shot 10 > outpaint_updown_sampling_shot10_artaxor.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python outpainting_updown_sampling_redux.py --process_id ${process_id} --dataset DIOR --shot 1 > ./log_outpaint/outpaint_updown_sampling_shot1_dior_process${process_id}.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python outpainting_updown_sampling_redux.py --process_id 1 --dataset DIOR --shot 5 > outpaint_updown_sampling_shot5_dior.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python outpainting_updown_sampling_redux.py --process_id 1 --dataset DIOR --shot 10 > outpaint_updown_sampling_shot10_dior.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python outpainting_updown_sampling_redux.py --process_id 1 --dataset NEU-DET --shot 5 > outpaint_updown_sampling_shot5_neu-det.log 2>&1 &
# CUDA_VISIBLE_DEVICE=5 nohup python outpainting_updown_sampling_redux.py --process_id 1 --dataset NEU-DET --shot 10 > outpaint_updown_sampling_shot10_neu-det.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python outpainting_updown_sampling_redux.py --process_id ${process_id} --dataset NEU-DET --shot 1 > ./log_outpaint/outpaint_updown_sampling_shot1_neu-det_process${process_id}.log 2>&1 &

# CUDA_VISIBLE_DEVICES=5 nohup python outpainting_updown_sampling_redux.py --process_id ${process_id} --dataset clipart1k --shot 1 > ./log_outpaint/outpaint_updown_sampling_shot1_clipart1k_process${process_id}.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python outpainting_updown_sampling_redux.py --process_id 1 --dataset clipart1k --shot 5 > outpaint_updown_sampling_shot5_clipart1k.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python outpainting_updown_sampling_redux.py --process_id 2 --dataset clipart1k --shot 10 > outpaint_updown_sampling_shot10_clipart1k.log 2>&1 &

# CUDA_VISIBLE_DEVICES=4 python outpainting_updown_sampling_redux.py --dataset ArTaxOr --shot 5 --failed_only --log_file outpaint_updown_sampling_shot5_artaxor.log > artaxor5.log 

# CUDA_VISIBLE_DEVICES=0 nohup python outpainting_updown_sampling_redux.py --dataset NWPU_VHR-10 --shot 3 > outpaint_updown_sampling_shot3_nwpu.log 2>&1 &
# # 处理5-shot
# CUDA_VISIBLE_DEVICES=0 nohup python outpainting_updown_sampling_redux.py --dataset NWPU_VHR-10 --shot 5 > outpaint_updown_sampling_shot5_nwpu_withplane.log 2>&1 &
# # 处理10-shot
# CUDA_VISIBLE_DEVICES=1 nohup python outpainting_updown_sampling_redux.py --dataset NWPU_VHR-10 --shot 10 > outpaint_updown_sampling_shot10_nwpu.log 2>&1 &
# # 处理20-shot
# CUDA_VISIBLE_DEVICES=5 nohup python outpainting_updown_sampling_redux.py --sample_id 548 --sample_dir /nvme/liyu/Flux/result/NWPU_VHR_10_20shot_retrieval/548 --dataset NWPU_VHR-10 --shot 20 > outpaint_updown_sampling_shot20_nwpu1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python outpainting_updown_sampling_redux.py --process_id 0 --dataset NWPU_VHR-10 --shot 20 > outpaint_updown_sampling_shot20_nwpu.log 2>&1 &

# 处理Camouflage数据集的不同shot数
# CUDA_VISIBLE_DEVICES=4 nohup python outpainting_updown_sampling_redux.py --dataset Camouflage --shot 1 --process_id ${process_id} > outpaint_updown_sampling_shot1_Camouflage.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python outpainting_updown_sampling_redux.py --dataset Camouflage --shot 2 --process_id ${process_id} > outpaint_updown_sampling_shot2_Camouflage.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python outpainting_updown_sampling_redux.py --dataset Camouflage --shot 3 --process_id ${process_id} > outpaint_updown_sampling_shot3_Camouflage.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python outpainting_updown_sampling_redux.py --dataset Camouflage --shot 5 --process_id ${process_id} > outpaint_updown_sampling_shot5_Camouflage.log 2>&1 &


