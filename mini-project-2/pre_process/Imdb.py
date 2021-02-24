from PreProcessor import PreProcessor, dataset_path
import pandas as pd


def convert_imbd_to_csv(file_lst, output_name):
    df = pd.DataFrame(columns=['review_id','train_or_test','review_type', 'review_number' ,'sentence'])

    for file in file_lst:
        with open(file, 'r') as f:
            detail = file.stem.split('_')
            path = str(file.parent).split('/')
            df = df.append({
                'train_or_test':path[1],'review_type':path[2], 'sentence':f.read(), 'review_number':detail[1], 'review_id':detail[0]
                }, ignore_index=True)
    df.to_csv(output_name, index=False)


imdb_raw_df = pd.read_csv(dataset_path.joinpath('imdb_raw.csv'))
"""
list of words that are common to both dataset
    > we can play with which word to remove and see the performance of the model
    br is a html tag
"""
common_words = [
    'br', 
    # 'film', 
    # 'movie', 
    # 'one', 
    # 'like', 
    # 'good',
    # 'time'
    ]
imdb_processor = PreProcessor(imdb_raw_df, common_words,'imdb')
imdb_processor.process()