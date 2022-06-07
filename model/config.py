from argparse import ArgumentParser, ArgumentTypeError
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


ZERO = 1e-8

INDEX_OF_LABEL = {'fake': 1, 'real': 0}
INDEX2LABEL = ['real', 'fake']

DATASETS_CHINESE = ['Weibo']
DATASETS_ENGLISH = ['Twitter']

MAX_TOKENS_OF_A_POST = 256

kernel_mu = np.arange(-1, 1.1, 0.1).tolist()
kernel_sigma = [20 for _ in kernel_mu]
kernel_mu.append(0.99)
kernel_sigma.append(100)


parser = ArgumentParser(description='NewsEnv4FEND')

# ======================== Dataset ========================

parser.add_argument('--dataset', type=str, default='Weibo')
parser.add_argument('--category_num', type=int, default=2)
parser.add_argument('--save', type=str, default='./ckpts/debug',
                    help='folder to save the final model')

parser.add_argument('--model', type=str,
                    help='[BERT, BiLSTM, EANN, DeClarE, MAC]')

# ======================== Framework ========================

parser.add_argument('--use_fake_news_detector', type=str2bool, default=True)

# --- MLP ---
parser.add_argument('--num_mlp_layers', type=int, default=3)

# --- News Env Features ---
parser.add_argument('--use_news_env', type=str2bool, default=True)
parser.add_argument('--use_macro_env', type=str2bool, default=True)
parser.add_argument('--use_micro_env', type=str2bool, default=True)
parser.add_argument('--use_semantics_of_news_env', type=str2bool, default=True)
parser.add_argument('--use_similarity_of_news_env',
                    type=str2bool, default=True)

parser.add_argument('--macro_env_days', type=int, default=3)
parser.add_argument('--micro_env_rate', type=float, default=0.1)
parser.add_argument('--micro_env_min_num', type=int, default=10)
parser.add_argument('--macro_env_output_dim', type=int, default=128)
parser.add_argument('--micro_env_output_dim', type=int, default=128)

# --- Fusion with the detection features ---
parser.add_argument('--strategy_of_fusion', type=str,
                    default='gate', help=['concat', 'att', 'gate'])
parser.add_argument('--multi_attention_dim', type=int, default=128)

# --- Sim Values MLP ---
parser.add_argument('--use_p_mac', type=str2bool, default=True)
parser.add_argument('--use_p_mic', type=str2bool, default=True)
parser.add_argument('--use_avgmic_mic', type=str2bool, default=True)

parser.add_argument('--sim_values_output_dim', type=int, default=512)


# ======================== Pattern-based Models ========================

# --- BiLSTM ---
parser.add_argument('--bilstm_input_max_sequence_length',
                    type=int, default=MAX_TOKENS_OF_A_POST//2)
parser.add_argument('--bilstm_input_dim', type=int, default=300)
parser.add_argument('--bilstm_hidden_dim', type=int, default=64)
parser.add_argument('--bilstm_num_layer', type=int, default=1)
parser.add_argument('--bilstm_dropout', type=float, default=0)

# --- EANN_Text ---
parser.add_argument('--eann_input_max_sequence_length',
                    type=int, default=MAX_TOKENS_OF_A_POST//2)
parser.add_argument('--eann_input_dim', type=int, default=300)
parser.add_argument('--eann_hidden_dim', type=int, default=64)
parser.add_argument('--eann_event_num', type=int, default=300)
parser.add_argument('--eann_weight_of_event_loss', type=float, default=-1.0)

# --- BERT ---
parser.add_argument('--bert_pretrained_model',
                    type=str, default='bert-base-chinese', help='[bert-base-chinese, bert-base-uncased]')
parser.add_argument('--bert_input_max_sequence_length',
                    type=int, default=MAX_TOKENS_OF_A_POST)
parser.add_argument('--bert_training_embedding_layers',
                    type=str2bool, default=True)
parser.add_argument('--bert_training_inter_layers',
                    type=str2bool, default=True)
parser.add_argument('--bert_hidden_dim', type=int, default=768)
parser.add_argument('--bert_use_emotion', type=str2bool, default=False)
parser.add_argument('--bert_emotion_features_dim', type=int, default=0)

# ======================== Fact-based Models ========================

parser.add_argument('--retrieve_ideally', type=str2bool, default=False,
                    help='`True` for retrieve all the articles without the limit of the time')
parser.add_argument('--relevant_articles_num', type=int, default=5)

# --- MAC ---
parser.add_argument('--mac_input_max_sequence_length',
                    type=int, default=MAX_TOKENS_OF_A_POST//2)
parser.add_argument('--mac_max_doc_length', type=int,
                    default=100)
parser.add_argument('--mac_input_dim', type=int, default=300)
parser.add_argument('--mac_hidden_dim', type=int, default=32)
parser.add_argument('--mac_dropout_doc', type=float, default=0.7)
parser.add_argument('--mac_dropout_query', type=float, default=0.1)
parser.add_argument('--mac_nhead_1', type=int, default=2)
parser.add_argument('--mac_nhead_2', type=int, default=2)

# --- DeClarE ---
parser.add_argument('--declare_input_max_sequence_lengtdh',
                    type=int, default=MAX_TOKENS_OF_A_POST//2)
parser.add_argument('--declare_input_dim', type=int, default=300)
parser.add_argument('--declare_hidden_dim', type=int, default=64)
parser.add_argument('--declare_max_doc_length', type=int,
                    default=100)
parser.add_argument('--declare_bilstm_num_layer', type=float, default=1)
parser.add_argument('--declare_bilstm_dropout', type=float, default=0.1)

# ======================== Training ========================

parser.add_argument('--lr', type=float, default=5e-5,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='No. of the epoch to start training')
parser.add_argument('--resume', type=str, default='',
                    help='path to load trained model')
parser.add_argument('--evaluate', type=str2bool, default=False,
                    help='only use for evaluating')
parser.add_argument('--inference_analysis', type=str2bool,
                    default=False, help='only use for inferencing')

parser.add_argument('--debug', type=str2bool, default=False)

# ======================== Devices ========================

parser.add_argument('--seed', type=int, default=9,
                    help='random seed')
parser.add_argument('--device', default='cpu')
parser.add_argument('--fp16', type=str2bool, default=True,
                    help='use fp16 for training')
parser.add_argument('--local_rank', type=int, default=-1)
