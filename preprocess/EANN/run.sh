# ===================== Configuration =====================
# 'Chinese' or 'English'
dataset_name='Chinese'
events_num=300

# ===================== Event Clustering by Kmeans =====================
python event_clustering.py --dataset ${dataset_name} --events_num ${events_num}