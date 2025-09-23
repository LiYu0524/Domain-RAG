output_dir=./results_neu
CUDA_VISIBLE_DEVICES=0 nohup python batch_generate_flux_kshot_neu.py --database neudet --dataset ArTaxOr --shots 1 --output_dir ${output_dir} > ${output_dir}_log_artax_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python batch_generate_flux_kshot_neu.py --database neudet --dataset FISH --shots 1 --output_dir ${output_dir} > ${output_dir}_log_fish_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python batch_generate_flux_kshot_neu.py --database neudet --dataset clipart1k --shots 1 --output_dir ${output_dir} > ${output_dir}_log_clipart_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python batch_generate_flux_kshot_neu.py --database neudet --dataset DIOR --shots 1 --output_dir ${output_dir} > ${output_dir}_log_dior_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python batch_generate_flux_kshot_neu.py --database neudet --dataset NEU-DET --shots 1 --output_dir ${output_dir} > ${output_dir}_log_neudet_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python batch_generate_flux_kshot_neu.py --database neudet --dataset UODD --shots 1 --output_dir ${output_dir} > ${output_dir}_log_uodd_1.log 2>&1 &