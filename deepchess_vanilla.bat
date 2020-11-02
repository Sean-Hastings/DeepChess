python parse_data.py --dataset ccrl
python train_autoencoder.py --dataset ccrl --batch_size 4096 --decay 0.1 --id overnight --epochs 3 --dropout 0.05
python featurize.py --dataset ccrl --model_id overnight
python train_siamese.py --dataset ccrl --batch_size 4096 --epochs 10000 --id overnight --dropout 0.05
