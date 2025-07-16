num_repeat=1
data="rsna"
method="ae-ssim"
gpu=0

# python train.py -d "$data" -m "$method" -g $gpu -f "$i" --input-size 128 -ls 64;
python test.py -d "$data" -m "$method" -g $gpu -f "$i" -save --input-size 128 -ls 64 \
  --test-model-path /home/goto/thesis/Experiment/MedIAnomaly/rsna2025-07-15_23-44-52/ae-ssim/fold_/checkpoints/model.pt;
  