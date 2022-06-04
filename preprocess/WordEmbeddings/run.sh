# ===================== Configuration =====================
# 'Chinese' or 'English'
dataset_name='Chinese'

# 'post' or 'article'
data_type='post'

# Config it as your local word-embeddings filepath
embedding_file='/data/zhangxueyao/word2vec/sgns.weibo.bigram-char'

# ===================== Tokenization by Word Embeddings =====================
python get_words_${dataset_name}.py --dataset ${dataset_name} --data_type ${data_type} --embedding_file ${embedding_file}