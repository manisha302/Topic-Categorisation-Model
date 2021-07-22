import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
import pandas as pd
from os import path
# from PIL import Image
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# import matplotlib.pyplot as plt
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.tag import pos_tag
nltk.download('inaugural')
from nltk.corpus import inaugural
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import pickle
import joblib
import Preprocess
# Run below comand if you are running IPython


# % matplotlib inline

stop_words=stopwords.words('english')
snowball_stemmer = SnowballStemmer('english')
wordnet_lemmatizer = WordNetLemmatizer()
df=pd.read_csv(r'bhim_raw_data.csv',encoding='utf-8')
Preprocessed_text=[]
Preprocessed_text = Preprocess.preprocess(df['Remarks'])
print(Preprocessed_text)


stop_words_new=['make','use','all','better','bhim','right','worst','thank','happy','like','awesome','give','good','bad','provide',
                'dont','doesnt''keep','please','would','send','often','thankyou','thanks','increase','unable','need',
                'atleast','easy','nothing','number','everything','excellent','raise','person','till','date','smooth',
                'without','with','kind','help','well','face','daily','quick','none','nice','always','want','sometime',
                'keep','others','work','available','able','miss','cant','take','life','also','bhim','very','didnt']


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
    
df=df.drop_duplicates(subset='Remarks',keep='first',inplace=False,ignore_index=True)
df['preprocess_text'] = df['Remarks'].apply(preprocess)
df['preprocess_text'] = df['preprocess_text'].apply(lambda x: x if len(x.split())>2 else None)
df['preprocess_text'] = df['preprocess_text'].replace(r'None', np.NaN, regex=True)
df=df.drop_duplicates(subset='preprocess_text',keep='first',inplace=False,ignore_index=True)
df = df.dropna( how='any',subset=['preprocess_text'])
df = df.reset_index(drop=True)
print(df)

# text= " ".join(review for review in df['preprocess_text'].astype(str))
# print ("There are {} words in the combination of all cells in column BLOOM.".format(len(text)))
# wordcloud = WordCloud(background_color="black", width=800, height=400).generate(text)

# plt.axis("off")
# plt.figure( figsize=(30,20))
# plt.tight_layout(pad=0)
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.show()
cv3 = pickle.load(open('preprocessing.pkl','rb'))
cv3(df['Remarks'])


vect = CountVectorizer()
X = vect.fit_transform(df['preprocess_text'])
pickle.dump(vect,open('Transform.pkl','wb'))

lda=LatentDirichletAllocation(max_iter=70,
                              n_components = 7,
                              learning_decay= .7,
                              learning_method='batch',
                              batch_size=1000,
                              random_state=100)
pickle.dump(lda,open('model.pkl','wb'))
lda_output = lda.fit_transform(X)
print(lda_output)