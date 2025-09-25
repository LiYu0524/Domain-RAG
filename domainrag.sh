# Domain-Aware Background Retrieval

# frist cd ./retrieval
CUDA_VISIBLE_DEVICES=4 nohup python clip100_resnet_style_all_shots.py --datasets NWPU_VHR_10 --shots 3 > log_retrival_vhr3.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python clip100_resnet_style_all_shots.py --datasets NWPU_VHR_10 --shots 5 > log_retrival_vhr5.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python clip100_resnet_style_all_shots.py --datasets NWPU_VHR_10 --shots 10 > log_retrival_vhr10.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python clip100_resnet_style_all_shots.py --datasets NWPU_VHR_10 --shots 20 > log_retrival_vhr20.log 2>&1 &


# Domain-Guided Background Generation

# cd path/to/Domain-RAG

output_dir=./results_neu
CUDA_VISIBLE_DEVICES=0 nohup python batch_generate_flux_kshot_neu.py --database neudet --dataset ArTaxOr --shots 1 --output_dir ${output_dir} > ${output_dir}_log_artax_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python batch_generate_flux_kshot_neu.py --database neudet --dataset FISH --shots 1 --output_dir ${output_dir} > ${output_dir}_log_fish_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python batch_generate_flux_kshot_neu.py --database neudet --dataset clipart1k --shots 1 --output_dir ${output_dir} > ${output_dir}_log_clipart_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python batch_generate_flux_kshot_neu.py --database neudet --dataset DIOR --shots 1 --output_dir ${output_dir} > ${output_dir}_log_dior_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python batch_generate_flux_kshot_neu.py --database neudet --dataset NEU-DET --shots 1 --output_dir ${output_dir} > ${output_dir}_log_neudet_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python batch_generate_flux_kshot_neu.py --database neudet --dataset UODD --shots 1 --output_dir ${output_dir} > ${output_dir}_log_uodd_1.log 2>&1 &

# Foreground-Background Composition

# process_id aims to save multiple result avoiding conflicts
process_id=1
CUDA_VISIBLE_DEVICES=0 nohup python outpainting_updown_sampling_redux.py --process_id ${process_id} --dataset clipart1k --shot 1 > ./log_outpaint/outpaint_updown_sampling_shot1_clipart1k_process${process_id}.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python outpainting_updown_sampling_redux.py --process_id ${process_id} --dataset UODD --shot 1 > ./log_outpaint/uodd_1shot_process${process_id}.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python outpainting_updown_sampling_redux.py --process_id ${process_id} --dataset FISH --shot 1 > ./log_outpaint/outpaint_updown_sampling_shot1_fish_process${process_id}_prompt.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python outpainting_updown_sampling_redux.py --process_id ${process_id} --dataset ArTaxOr --shot 1 > ./log_outpaint/outpaint_updown_sampling_shot1_artaxor_process${process_id}.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python outpainting_updown_sampling_redux.py --process_id ${process_id} --dataset DIOR --shot 1 > ./log_outpaint/outpaint_updown_sampling_shot1_dior_process${process_id}.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python outpainting_updown_sampling_redux.py --process_id ${process_id} --dataset NEU-DET --shot 1 > ./log_outpaint/outpaint_updown_sampling_shot1_neu-det_process${process_id}.log 2>&1 &
