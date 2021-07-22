import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
import pandas as pd
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.tag import pos_tag
nltk.download('inaugural')
from nltk.corpus import inaugural
stop_words=stopwords.words('english')
snowball_stemmer = SnowballStemmer('english')
wordnet_lemmatizer = WordNetLemmatizer()

stop_words_new=['make','use','all','better','bhim','right','worst','thank','happy','like','awesome','give','good','bad','provide',
                'dont','doesnt''keep','please','would','send','often','thankyou','thanks','increase','unable','need',
                'atleast','easy','nothing','number','everything','excellent','raise','person','till','date','smooth',
                'without','with','kind','help','well','face','daily','quick','none','nice','always','want','sometime',
                'keep','others','work','available','able','miss','cant','take','life','also','bhim','very','didnt']

df=pd.read_csv(r'bhim_raw_data.csv',encoding='utf-8')

def preprocess(text1):
    
    text1 = str(text1)
    clean_text1 = re.sub('[^a-zA-z]',' ',str(text1))
    clean_text1 = word_tokenize(clean_text1)
    clean_text1 = [str(i).replace(r'D_\n', ' ') for i in clean_text1]
    clean_text1 = [str(i).replace(r'_x', '') for i in clean_text1]
#     clean_text1 = word_tokenize(clean_text1)
    clean_text1 = [i.lower() for i in clean_text1 if i not in string.punctuation]
    clean_text1 = [wordnet_lemmatizer.lemmatize(word,pos="v") for word in clean_text1]
    clean_text1 = [wordnet_lemmatizer.lemmatize(word,pos="n") for word in clean_text1]
    clean_text1 = [wordnet_lemmatizer.lemmatize(word,pos="r") for word in clean_text1]
    clean_text1 = [i for i in clean_text1 if not i in stop_words and not i in stop_words_new]
    clean_text1 = nltk.pos_tag(clean_text1)
    clean_text1 = [word for (word,pos) in clean_text1 if (pos != 'IN' or pos != 'DT' or pos != 'PRP'  or pos != 'MD' or pos !='PRP$')]
#     clean_text1 = " ".join([i for i in clean_text1 if not i in stop_words_new])
    clean_text1 = " ".join([i for i in clean_text1 if len(i)>3])
    return(clean_text1)
    
# df = df.drop_duplicates(subset='Remarks',keep='first',inplace=False,ignore_index=True)
# df['preprocess_text'] = df['Remarks'].apply(preprocess)
# df['preprocess_text'] = df['preprocess_text'].apply(lambda x: x if len(x.split())>2 else None)
# df['preprocess_text'] = df['preprocess_text'].replace(r'None', np.NaN, regex=True)
# df = df.drop_duplicates(subset='preprocess_text',keep='first',inplace=False,ignore_index=True)
# df = df.dropna( how='any',subset=['preprocess_text'])
# df = df.reset_index(drop=True)
# print(df)

import pickle
pickle.dump(preprocess,open('preprocessing.pkl','wb'))