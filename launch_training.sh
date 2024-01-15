export CUDA_VISIBLE_DEVICES=3
export NGPUS=1
export OMP_NUM_THREADS=8 # you can change this value according to your number of cpu cores


# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --resume ./logs/20231206_103047_lr_1e-01_b_8/ep056.pth --config configs/chezai_lane.py 
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py  --config configs/chezai_lane.py 
