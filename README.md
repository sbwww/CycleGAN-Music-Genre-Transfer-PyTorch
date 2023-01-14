# Symbolic Music Genre Transfer with CycleGAN

PyTorch 1.10 复现

参考资料

- [sumuzhao/CycleGAN-Music-Style-Transfer](https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer)：论文仓库，TensorFlow 实现
- [Asthestarsfalll/Symbolic-Music-Genre-Transfer-with-CycleGAN-for-pytorch](https://github.com/Asthestarsfalll/Symbolic-Music-Genre-Transfer-with-CycleGAN-for-pytorch)：论文 PyTorch 复现，但是 loss 计算和 backward 过程存在问题，无法运行在 1.5 及以上的 PyTorch，See [RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation? #39141](https://github.com/pytorch/pytorch/issues/39141#issuecomment-636881953)
- [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)：CycleGAN 论文仓库

## 1. 准备环境和数据

```sh
pip install -r requirements.txt
unzip -d ./data/ Dataset.zip
```

## 2. 训练

修改 `train.sh`，例如

```sh
export CUDA_VISIBLE_DEVICES=0

MODEL_NAME=CP
EPOCH=30
BS=4

python train.py \
    --epoch ${EPOCH} \
    --batch_size ${BS} \
    --data_mode partial \
    --sigma 0.1 \
    --model_name ${MODEL_NAME} \
    --save_frq 5000 \
    --log_frq 5000
```

具体的各参数可以使用 `python train.py -h` 查看

运行

```sh
bash train.sh
```

训练后的模型将保存在 `saved_models` 文件夹中

## 3. 推断

修改 `test.sh`，例如

```sh
export CUDA_VISIBLE_DEVICES=0

MODEL_NAME=CP

python test.py \
    --model_dir ./saved_models/${MODEL_NAME}/final.pth \
    --batch_size 1 \
    --model_name ${MODEL_NAME} \
    --test_mode A2B
```

具体的各参数可以使用 `python test.py -h` 查看

```sh
bash test.sh
```

转换的 MIDI 文件将保存在 `test` 文件夹中
