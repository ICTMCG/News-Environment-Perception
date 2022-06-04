# ===================== Configuration =====================
# 'Chinese' or 'English'
dataset_name='Chinese'
# 'bert-base-chinese' or 'bert-base-uncased'
pretrained_model='bert-base-chinese'

# ===================== Tokenization by BERT =====================
python get_bert_tokens.py --dataset ${dataset_name} --pretrained_model ${pretrained_model}