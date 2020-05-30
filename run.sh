source activate torch1.0
export CUDA_VISIBLE_DEVICES=0,1 && nohup python -u train.py --arch mobilenetv2 --checkpoints ./models > a0.log 2>&1 &
tail -f a0.log
