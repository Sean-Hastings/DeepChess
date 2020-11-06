python parse_data.py --dataset ccrl
python train_autoencoder.py --dataset ccrl --batch_size 4096 --decay 0.1 --id overnight
python train_siamese.py --dataset ccrl --batch_size 4096 --epochs 1000 --id overnight --ae_source overnight
