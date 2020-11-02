python parse_data.py --dataset ccrl
python train_autoencoder.py --batch_size 4096 --decay 0.1 --id overnight --epochs 3
python featurize.py
python train_siamese.py --batch_size 4096 --epochs 10000 --id overnight
