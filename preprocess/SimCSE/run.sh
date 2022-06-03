# ===================== Configuration =====================
dataset_name='Chinese'
# dataset_name='English'

# ===================== Obtain the representations of posts and news =====================
CUDA_VISIBLE_DEVICES=0 python get_repr.py --dataset ${dataset_name}