export CUDA_VISIBLE_DEVICES=1

MODEL_NAME=CP

python test.py \
    --model_dir ./saved_models/${MODEL_NAME}/final.pth \
    --batch_size 1 \
    --model_name ${MODEL_NAME} \
    --test_mode A2B
