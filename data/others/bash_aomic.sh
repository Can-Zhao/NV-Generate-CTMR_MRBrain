# Step 1: Apply VAE (run with torchrun for multi-GPU)
torchrun --nnodes=1 --nproc_per_node=8 \
  ./data/others/3_apply_vae.py \
  --dataset-name aomic \
  --input-json /lustre/fsw/portfolios/healthcareeng/users/canz/code/jsons/dataset_aomic_0.json \
  --input-root /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/monai_datasets/aomic_skull_stripped \
  --output-root /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/MAISI/data/encoding_128_downsample/aomic_skull_stripped/ \
  --num-splits 1

# Step 2: Create JSON
python ./data/others/4_create_json_mr_emb.py \
  --dataset-name aomic \
  --input-json /lustre/fsw/portfolios/healthcareeng/users/canz/code/jsons/dataset_aomic_0.json \
  --input-root /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/monai_datasets/aomic_skull_stripped \
  --embedding-root /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_monai/MAISI/data/encoding_128_downsample/aomic_skull_stripped/ \
  --output-json /lustre/fsw/portfolios/healthcareeng/users/canz/code/NV-Generate-CTMR_MRbrain/jsons/dataset_aomic_0_emb.json \