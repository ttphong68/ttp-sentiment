#----------------------------------------------------------------------------------------------------
#### HV : THÁI THANH PHONG
#### SN : 14/03/1968
#### ĐỒ ÁN TỐT NGHIỆP : Project 1 Data Prepreocessing
#### MÔN HỌC : Data science
#----------------------------------------------------------------------------------------------------
# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
import datetime
import streamlit as st

from underthesea import word_tokenize
import glob
from wordcloud import WordCloud,STOPWORDS
# from pandas_profiling import ProfileReport
# import scipy

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
# !pip install import-ipynb
import import_ipynb
from Library_Functions import *
import pickle

#----------------------------------------------------------------------------------------------------
# Part 1: Build project
# load data
print('Loading data.....')
df = pd.read_csv('data/Products_Shopee_comments_cleaned_ver01.csv')

# abspath1 = os.path.abspath('data/Products_Shopee_comments_cleaned_ver01_1.csv')
# df1=pd.read_csv(abspath1,sep=',', encoding='utf-8')
# abspath2 = os.path.abspath('data/Products_Shopee_comments_cleaned_ver01_2.csv')
# df2=pd.read_csv(abspath2,sep=',', encoding='utf-8')
# df = pd.concat([df1, df2], axis=0)

# Data pre - processing
df.drop(['Unnamed: 0'],axis=1,inplace=True)
# remove duplicate
df.drop_duplicates(inplace=True)
# remove missing values
df.dropna(inplace=True)

# Đọc model LogisticRegression()
pkl_filename = "model/Sentiment_model_best.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

# Đọc model CountVectorizer()
filename = "model/cv_model.pkl"
with open(filename, 'rb') as file:
    cv = pickle.load(file)

# split data into train and test
print('Split data into train and test...........')
X=df['comment']
y=df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_cv = cv.transform(X_test)
y_pred = model.predict(X_test_cv)
cm = confusion_matrix(y_test, y_pred)

##LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
##LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('files/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('files/wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

# Part 2: Build app

# Title
st.image('download.jpg')

st.title("Trung tâm tin học - ĐH KHTN")
st.header('Data Science and Machine Learning Certificate')
st.subheader('Sentiment analysis of Vietnamese comments on Shopee')

# st.video('https://www.youtube.com/watch?v=q3nSSZNOg38&list=PLFTWPHJsZXVVnckL0b3DYmHjjPiRB5mqX')

menu = ['Tổng quan', 'Xây dựng mô hình', 'Dự đoán mới']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Tổng quan':
    st.subheader('Tổng quan')
    
    st.write('''
    Đây là dự án về phân tích cảm xúc của các bình luận của người Việt trên Shopee.
    Bộ dữ liệu được thu thập từ trang web Shopee (Do Cô cung cấp).
    Tập dữ liệu chứa 2 cột: xếp hạng và bình luận.
    Thang đánh giá là từ 1 đến 5.
    Cột Comment là nhận xét của khách hàng sau khi họ mua hàng trên Shopee.
    Mục tiêu của dự án này là xây dựng một mô hình để dự đoán cảm xúc của các bình luận.
    Cảm xúc tích cực hay tiêu cực.
    ''')
    st.write('''
    Bộ dữ liệu có 2 lớp: Tích cực và tiêu cực. 
    ''')
    st.write('''
    Mô hình được xây dựng với Logistic Regression và sử dụng oversampling data:
    - Mô hình có độ chính xác 87% ( Accuracy ).
    - Mô hình có độ chính xác 94% ( precision ) cho lớp tích cực ( positive class ).
    - Mô hình có độ chính xác 85% ( recall ) cho lớp tích cực ( positive class ).
    - Mô hình có độ chính xác 71% ( precision ) cho lớp tiêu cực ( negative class ).
    - Mô hình có độ chính xác 87% ( recall ) cho lớp tiêu cực ( negative class ). 
    ''')
elif choice == 'Xây dựng mô hình':
    st.subheader('Xây dựng mô hình')
    st.write('#### Tiền xử lý dữ liệu')
    st.write('##### Hiển thị dữ liệu')
    st.table(df.head())
    # plot bar chart for sentiment
    st.write('##### Biểu đồ Bar cho biểu thị tình cảm')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df['Sentiment'].value_counts().index, df['Sentiment'].value_counts().values)
    ax.set_xticks(df['Sentiment'].value_counts().index)
    ax.set_xticklabels(['Tiêu cực', 'Tích cực'])
    ax.set_ylabel('Số lượng bình luận')
    ax.set_title('Biểu đồ Bar cho biểu thị tình cảm')
    st.pyplot(fig)

    # plot wordcloud for positive and negative comments
    st.write('##### Wordcloud Cho bình luận tích cực')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['Sentiment'] == 1]['comment'])))
    ax.axis('off')
    st.pyplot(fig)

    st.write('##### Wordcloud Cho bình luận tiêu cực')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['Sentiment'] == 0]['comment'])))
    ax.axis('off')
    st.pyplot(fig)

    st.write('#### Xây dựng mô hình và đánh giá:')
    st.write('##### Confusion matrix')
    st.table(cm)
    st.write('##### Classification report')
    st.table(classification_report(y_test, y_pred, output_dict=True))
    st.write('##### Accuracy')
    # show accuracy as percentage with 2 decimal places
    st.write(f'{accuracy_score(y_test, y_pred) * 100:.2f}%')
    
elif choice == 'Dự đoán mới':
    st.subheader('Dự đoán mới')
    st.write('''
    Nhập vào một bình luận và mô hình sẽ dự đoán tình cảm của bình luận. 
    ''')
    comment = st.text_input('Nhập vào một bình luận')
    if st.button('Dự đoán'):
        if comment != '':

            # # Xử lý tiếng việt thô
            comment = process_text(comment, emoji_dict, teen_dict, wrong_lst)
            # Chuẩn hóa unicode tiếng việt
            comment = covert_unicode(comment)
            # Kí tự đặc biệt
            comment = process_special_word(comment)
            # postag_thesea
            comment = process_postag_thesea(comment)
            #  remove stopword vietnames
            comment = remove_stopword(comment, stopwords_lst)

            comment = cv.transform([comment])
            y_predict = model.predict(comment)

            if y_predict[0] == 1:
                st.write('Tình cảm của bình luận là tích cực')
            else:
                st.write('Tình cảm của bình luận là tiêu cực')
        else:
            st.write('Nhập vào một bình luận')