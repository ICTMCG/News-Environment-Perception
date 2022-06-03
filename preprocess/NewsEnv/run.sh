# ===================== Configuration =====================
dataset_name='Chinese'
# dataset_name='English'
macro_env_days=3

# ===================== Get the macro env and rank its internal items by similarites =====================
python get_env.py --dataset ${dataset_name} --macro_env_days ${macro_env_days}