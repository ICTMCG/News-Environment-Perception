import imp
import json
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import random
from argparse import ArgumentParser
import os
from sklearn.cluster import KMeans
import jieba


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--events_num', type=int, default=300)
    args = parser.parse_args()

    experimental_dataset = args.dataset
    events_num = args.events_num

    save_dir = 'data'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, experimental_dataset)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    datasets = []

    for t in ['train', 'val', 'test']:
        with open('../../dataset/{}/post/{}.json'.format(experimental_dataset, t), 'r') as f:
            pieces = json.load(f)
            for p in tqdm(pieces):
                p['words'] = list(jieba.cut(p['content']))
            print(len(pieces))
            datasets.append(pieces)

    corpus = [p['words'] for pieces in datasets for p in pieces]
    corpus = [' '.join(words) for words in corpus]
    if experimental_dataset == 'English':
        corpus = [t.lower() for t in corpus]
    print('Corpus: ', len(corpus))

    print('Eg: {}'.format(corpus[0]))

    # ============= Get TF-IDF =============
    vectorizer = CountVectorizer(max_features=6000)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    print(len(word), weight.shape)

    # with open('data/{}/TFIDF.pkl'.format(experimental_dataset), 'wb') as f:
    #     pickle.dump([word, weight], f)

    # ============= Kmeans =============
    clf = KMeans(events_num, verbose=True)
    clf.fit(weight)

    with open('./data/{}/Kmeans.pkl'.format(experimental_dataset), 'wb') as f:
        pickle.dump(clf, f)

    centers = clf.cluster_centers_
    labels = clf.labels_
    print(centers.shape, labels.shape)

    # See some samples
    for _ in range(3):
        print('*'*30, '\n')
        c = random.randint(0, events_num-1)
        arr = np.where(labels == c)[0]
        print('class: {}, sz: {}\n'.format(c, len(arr)))

        for idx in arr:
            print('-'*15, idx, '-'*15)
            print(corpus[idx])
            print()

    train_sz = len(datasets[0])
    val_sz = len(datasets[1])
    test_sz = len(datasets[2])

    np.save('./data/{}/train_event_labels.npy'.format(experimental_dataset),
            labels[:train_sz])
    np.save('./data/{}/val_event_labels.npy'.format(experimental_dataset),
            labels[train_sz:train_sz+val_sz])
    np.save('./data/{}/test_event_labels.npy'.format(experimental_dataset),
            labels[train_sz+val_sz:])
