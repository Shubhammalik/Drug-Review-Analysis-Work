"""
# Dataset Link
# https://www.kaggle.com/jessicali9530/kuc-hackathon-winter-2018

To avoid certification error update the download link to your kaggle signed in account
"""
import pathlib
import urllib
import pandas as pd
from zipfile import ZipFile


def download_dataset():

    url = "https://storage.googleapis.com/kaggle-data-sets/76158/171475/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1584237046&Signature=ODlyNzsQ2haGkT0HeRegpxUrbKc4kwPy9TLkNiTF8sZJRdIG%2F9sgK%2Bkv%2BNttVmoL8JUrjKAvn2EwWwmue%2FnOhieuc%2BEb2sMo376nPbVg3BJ1w%2BvSMO4i3xWQLtQ9%2BZRzucSUC3m5k2buL%2FNhJrL5LQCr4WSiDk1VFksZUbJ09uajacQVJWlbU6hfqFjps3nM2KrR3G8hGIgX2n4Tne1aImHgIzswWSPZU74hnqpv5kkxxQdDXtbfhhcQ7V9D92HQCFXRFWNQbUQVfx1TkuH2VEtPLE3PYoRYrEh9Oxr%2Fse2KtmnMzaj3OB5S9Z7mDNn%2FptMkJDI3puir0A2%2BG6CINg%3D%3D&response-content-disposition=attachment%3B+filename%3Dkuc-hackathon-winter-2018.zip"

    file_name = 'kuc-hackathon-winter-2018.zip'
    print('Downloading dataset')
    urllib.request.urlretrieve(url, file_name)


def read_zip():
    print('Reading multiples files')
    zip_file = ZipFile('kuc-hackathon-winter-2018.zip')
    df_train = pd.read_csv(zip_file.open('drugsComTrain_raw.csv'))
    df_test = pd.read_csv(zip_file.open('drugsComTest_raw.csv'))
    return df_train, df_test

def combine_datasets(df_train, df_test):
    print('Combining datasets')
    combined_frames = [df_train, df_test]
    drugs_dot_com = pd.concat(combined_frames)
    drugs_dot_com.columns = ['user_id', 'drug', 'condition', 'review', 'rating', 'date', 'useful_count']
    pathlib.Path('./../../../dataset').mkdir(exist_ok=True)
    pathlib.Path('./../../../dataset/drugs_com').mkdir(exist_ok=True)
    drugs_dot_com.to_csv('./../../../dataset/drugs_com/drugs_dot_com.csv', index=False)


def obtain_dataset():
    download_dataset()
    df_train, df_test = read_zip()
    combine_datasets(df_train, df_test)


obtain_dataset()