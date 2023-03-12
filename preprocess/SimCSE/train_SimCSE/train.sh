##### Chinese data #####
# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --train_file ./data/Chinese_train.txt \
#     --max_length 256 \
#     --pretrained bert-base-chinese \
#     --learning_rate 5e-6 \
#     --save_final True \
#     --tau 0.05 \

##### English data #####
CUDA_VISIBLE_DEVICES=0 python train.py \
    --train_file ./data/English_train.txt \
    --max_length 128 \
    --pretrained bert-base-uncased \
    --learning_rate 5e-6 \
    --dropout_rate 0.1 \
    --save_final True \
    --tau 0.05 \
