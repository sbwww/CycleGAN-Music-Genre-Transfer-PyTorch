export CUDA_VISIBLE_DEVICES=2

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
