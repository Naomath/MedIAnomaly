num_repeat=1
data="rsna"
method="ae"
gpu=0

python test.py -d "$data" -m "$method" -g $gpu -f "$i" -save --input-size 128 -ls 16 \
  --test-model-path /home/goto/thesis/Experiment/MedIAnomaly/rsna2025-07-16_01-02-31/ae/fold_/checkpoints/model.pt \
  --laplace --epsilon 100 --sensitivity-path /home/goto/thesis/Experiment/MedIAnomaly/rsna2025-07-16_01-02-31/ae/fold_/test_results/latent_range.csv;
  