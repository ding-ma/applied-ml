from PreProcessor import PreProcessor, dataset_path
from sklearn.datasets import fetch_20newsgroups
import pandas as pd


twenty_news_group_train  = fetch_20newsgroups(subset='train', remove=(['headers', 'footers', 'quotes']))
twenty_news_group_test  = fetch_20newsgroups(subset='test', remove=(['headers', 'footers', 'quotes']))

twenty_news_train_df = pd.DataFrame(data={'target': twenty_news_group_train.target, 'train_or_test':'train', 'sentence': twenty_news_group_train.data})
twenty_news_test_df = pd.DataFrame(data={'target': twenty_news_group_test.target, 'train_or_test':'test', 'sentence': twenty_news_group_test.data})
twenty_news_combined_df = twenty_news_train_df.append(twenty_news_test_df)


twenty_news_combined_df['sentence'] = twenty_news_combined_df['sentence'].apply(lambda x: x.replace('\n', ' ').replace('\r', '').replace('\t', ' ').strip())
twenty_news_combined_df.reset_index(inplace=True)
twenty_news_combined_df.rename(columns={'index': 'id'}, inplace=True)
twenty_news_combined_df.to_csv(dataset_path.joinpath('twenty_news_raw.csv'), sep='\t', index=False)


twenty_news_raw_df = pd.read_csv(dataset_path.joinpath('twenty_news_raw.csv'), sep='\t')
common_words = [
    # TODO: to be determined
]

twenty_news_processor = PreProcessor(twenty_news_raw_df,common_words,'twenty_news')
twenty_news_processor.process()