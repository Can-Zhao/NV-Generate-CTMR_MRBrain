# Step 1: Apply VAE (run with torchrun for multi-GPU)
torchrun --nnodes=1 --nproc_per_node=2 \
  ./data/others/3_apply_vae.py \
  --input_json /lustre/fsw/portfolios/healthcareeng/users/canz/code/jsons/dataset_ISLES2022_0.json \
  --data_root /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/monai_datasets/ISLES2022 \
  --output_root /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/MAISI/data/encoding_128_downsample/ISLES2022/ \
  --num-splits 1

# Step 2: Create JSON
python ./data/others/4_create_json_mr_emb.py \
  --dataset-name ISLES2022 \
  --input-json /lustre/fsw/portfolios/healthcareeng/users/canz/code/jsons/dataset_ISLES2022_0.json \
  --input-root /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/monai_datasets/ISLES2022 \
  --embedding-root /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/MAISI/data/encoding_128_downsample/ISLES2022/ \
  --output-json /lustre/fsw/portfolios/healthcareeng/users/canz/code/NV-Generate-CTMR_MRbrain/jsons/dataset_ISLES2022_0_emb.json
