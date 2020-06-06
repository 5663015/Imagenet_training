# Imagenet_training
## 训练

```
export CUDA_VISIBLE_DEVICES=6,7 && nohup python -u train.py \
	--arch model_name \
	--lr 0.05 \
	--weight_decay 4e-5 \
	--checkpoint your_path > xx.log 2>&1 &
```

## 实验模型与结果

| 模型            | accuracy (top-1) / % | running time (2 V100 GPUs) |
| --------------- | -------------------- | -------------------------- |
| EfficientNet_b0 | 74.1                 | about 66 hours             |
| MixNet_m        | 73.15                | about 92 hours             |
| MobileNet_v3    | 72.042               | about 52 hours             |
| MNASNet_a1      | 72.342               | about 51 hours             |





