source activate torch1.0
export CUDA_VISIBLE_DEVICES=0 && nohup python -u imagenet.py -a mobilenetv2 \
    -b 256 \
    -d /home/work/dataset/ILSVRC2012 \
    --epochs 150 \
    --lr-decay cos \
    --lr 0.05 \
    --wd 4e-5 \
    -c ../models \
    --width-mult 1.0 \
    --input-size 224 \
    -j 32 > a0.log 2>&1 &
tail -f a0.log
