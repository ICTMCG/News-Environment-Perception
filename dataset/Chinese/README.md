# Chinese Dataset

## Access

You will be shared the dataset by email after an ["Application to Use the Datasets for XXXXXX"]() has been submitted.

## Description

### Post

The posts are saved in `post` folder, which contain `train.json`, `val.json`, and `test.json`. In every json file:

- the `content` identifies the content of the post.
- the `label` identifies the veracity of the post, whose value is `fake` or `real`.
- the `time` identifies the publised time of the post.

### News Environment

The news environment items are saved in `news/news.json`. In the file:

- the `id` identifies the unique id of the news.
- the `content` identifies the content of the news.
- the `time` identifies the publised time of the news.

### Relevant Articles

The relevant articles are saved in `articles/articles.json`. In the file:

- the `_id` (if exists) identifies the unique id of the article.
- the `time` and `time_format` (if exists) identifies the publised time of the article.
- the `content` identifies the content of the article.
- the `content_all` TODO!!!!!!