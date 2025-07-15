num_repeat=1
data="rsna"
method="ae"
gpu=0

python train.py -d "$data" -m "$method" -g $gpu -f "$i";
python test.py -d "$data" -m "$method" -g $gpu -f "$i" -save;