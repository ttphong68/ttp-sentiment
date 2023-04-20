### import libraries ###

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

import scipy
from scipy.stats import iqr
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from underthesea import word_tokenize, pos_tag, sent_tokenize
import regex
import demoji
from pyvi import ViPosTagger, ViTokenizer
import string

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

# for report:
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# vẽ đường cong ROC
from sklearn.metrics import roc_auc_score, roc_curve

import streamlit as st
import pickle

import warnings
warnings.filterwarnings("ignore")

import re
import os
import datetime
import streamlit as st
import io

import glob
from wordcloud import WordCloud,STOPWORDS

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import speech_recognition as sr
from gtts import gTTS
import playsound
import pyaudio
from datetime import date,datetime
import datetime
from time import strftime
import os
import time
import sys
import ctypes
import json
import re
import webbrowser
import smtplib
import requests
import urllib
import urllib.request as urllib2

import warnings
warnings.filterwarnings("ignore")
#----------------------------------------------------------------------------------------------------
# import openai
# # link tạo API Key đó https://beta.openai.com/account/api-keys
# openai.organization = 'org-Vf0cOHTHl3VyD7bUQSqDmglv'
# # link lấy Organization ID https://beta.openai.com/account/org-settings
# openai.api_key = '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'
#------------------------------------------------------------------------------------------------------------------
def recognize_speech_voice():

    # Khởi tạo micro
    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        # Điều chỉnh độ lớn của micro nếu cần thiết
        r.adjust_for_ambient_noise(source)
        st.write("Đang ghi âm...")
        audio = r.listen(source)
        st.write("Đã ghi âm xong.")
    
    # Lưu file audio vào bộ nhớ tạm (buffer) là một object io.BytesIO
    audio_file = io.BytesIO(audio.get_raw_data())
    st.write("Đã lưu file audio vào bộ nhớ tạm.")
    
    # Nhận dạng giọng nói và chuyển đổi thành văn bản
    try:
        st.write("Đang nhận dạng giọng nói...")
        text = r.recognize_google(audio, language='vi-VN')
        # st.write(f"Bạn đã bình luận: {text}")
    except sr.UnknownValueError:
        st.write("Không thể nhận dạng giọng nói")
    except sr.RequestError as e:
        st.write(f"Lỗi trong quá trình kết nối tới Google Speech Recognition service: {e}")
    return text  
#------------------------------------------------------------------------------------------------------------------
def dung():
    text_to_speech("Hẹn gặp lại bạn sau!")
#------------------------------------------------------------------------------------------------------------------
def nhan_text():
    for i in range(3):
        st.write('Mời bạn bình luận: ')
        text = speech_to_text()
        if text:
            return text.lower()
        elif i < 2:
            speech_to_text("Tôi không nghe rõ. Bạn nói lại được không!")
    time.sleep(2)
    st.write("Hẹn gặp lại bạn sau!")
    dung()
    return 0
## end voice
#---------------------------------------------------------------------------------------------
def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        # Giảm tạp âm
        r.adjust_for_ambient_noise(source)
        # audio = r.listen(source,phrase_time_limit=5)
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio,language="vi-VI")
    except:
        text="Xin lỗi! tôi không nhận được giọng nói"
    return str(text)
#---------------------------------------------------------------------------------------------
def text_to_speech(text):
    output = gTTS(text,lang="vi", slow=False)
    date_string = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    filename = "voice"+date_string+".mp3"
    output.save(filename)
    playsound.playsound(filename)
    os.remove(filename)
#---------------------------------------------------------------------------------------------
def chatGPT(you):
    # List questions
    questions = [you]
    for question in questions:
        print('\r\n' + question)
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=question,
            temperature=0,
            max_tokens=512,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        print(' => ' + response.choices[0].text[2:])
        return response.choices[0].text[2:]
#---------------------------------------------------------------------------------------------

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

### FUNCTIONS ###
#---------------------------------------------------------------------------------------------
@st.cache_data
def df():
    df = pd.read_csv('processed_data.csv')
    return df
#---------------------------------------------------------------------------------------------
# load data parquet
@st.cache_data
def df_parquet():
    df = pd.read_parquet('processed_data.parquet')
    return df
#---------------------------------------------------------------------------------------------
@st.cache_data
def load_model_Sentiment_1():
    # load model finalised_model.sav
    filename = 'finalized_model.sav'        # Model của Long
    # filename = 'Sentiment_model_best.pkl' # Model của Phong
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model
#---------------------------------------------------------------------------------------------
@ st.cache_data
def load_model_tf():
    filename = "tfidf.pkl"          # tfidfVectorizer 
    # filename = "cv_model.pkl"     # CountVectorizer()
    with open(filename, 'rb') as file:
        tf = pickle.load(file)
    return tf
#---------------------------------------------------------------------------------------------
## Hàm kiểm tra và tính số lượng, tỷ trọng outliers
def check_outlier(col):
    Q1 = np.percentile(col, 25)
    print('Q1:       ', Q1)
    Q3 = np.percentile(col, 75)
    print('Q3:       ', Q3)
    IQR = scipy.stats.iqr(col)
    print('IQR:      ', IQR)
    highOutliers = (col >= Q3 + 1.5*IQR).sum()
    lowOutliers  = (col <= Q1 - 1.5*IQR).sum()
    print('# Number of upper outliers: ', highOutliers)
    print('# Number of lower outliers: ', lowOutliers)
    print('# Percentage of ouliers:    ', (highOutliers + lowOutliers)/col.shape[0])
#---------------------------------------------------------------------------------------------
## Hàm remove outliers
def remove_outlier(variable, data_param):
# Detection
    Q1 = np.percentile(data_param[variable], 25)
    Q3 = np.percentile(data_param[variable], 75)
    IQR = scipy.stats.iqr(data_param[variable])
    # Upper bound
    upper = np.where(data_param[variable] >= (Q3 + 1.5*IQR))
    # Lower bound
    lower = np.where(data_param[variable] <= (Q1 - 1.5*IQR))
    # Removing the Outliers
    data_param.drop(upper[0], inplace = True)
    data_param.drop(lower[0], inplace = True)
    data_param.reset_index(drop=True, inplace=True)
    return data_param
#---------------------------------------------------------------------------------------------
def process_text(text, emoji_dict, teen_dict, wrong_lst):
    document = text.lower()
    document = document.replace("’", '')
    document = regex.sub(r'\.+', ".", document)
    # Remove punctuation
    document = regex.sub('[^\w\s]', ' ', document)
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for char in punctuation:
        document = document.replace(char, ' ')

    # Remove numbers, only keep letters
    document = regex.sub(r'[\w]*\d+[\w]*', "", document) # document.replace('[\w]*\d+[\w]*', '', regex=True)

    # Some lines start with a space, remove them
    document = regex.sub('^[\s]{1,}', '', document)

    # # Remove multiple spaces with one space
    document = regex.sub('[\s]{2,}', ' ', document)

    # Some lines end with a space, remove them
    document = regex.sub('[\s]{1,}$', '', document)

    # Remove end of line characters
    document = regex.sub(r'[\r\n]+', ' ', document)

    # Remove HTTP links
    document = regex.sub(
        r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', '',
        document)

    new_sentence = ''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word] + ' ' if word in emoji_dict else word for word in list(sentence))
        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern, sentence))
        ###### DEL wrong words
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence + sentence + '. '
    document = new_sentence
    # print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    # ...
    return document
#---------------------------------------------------------------------------------------------
# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic
#---------------------------------------------------------------------------------------------
# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)
#---------------------------------------------------------------------------------------------
def process_special_word(text):
    new_text = ''
    text_lst = text.split()
    i= 0
    if 'không' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  word == 'không':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()
#---------------------------------------------------------------------------------------------
def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.','')
        ###### POS tag
        lst_word_type = ['N','Np','A','AB','V','VB','VY','R']
        # lst_word_type = ['A','AB','V','VB','VY','R']
        sentence = ' '.join( word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document
#---------------------------------------------------------------------------------------------
def remove_stopword(text, stopwords):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document
#---------------------------------------------------------------------------------------------
def text_transform(comment):
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
    return comment 