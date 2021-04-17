from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
from string import digits
import spacy
import pandas as pd
import time
from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# only run these once 
# nltk.download('punkt')
# nltk.download('stopwords')


def remove_ints(text):
    # https://stackoverflow.com/questions/12851791/removing-numbers-from-string
    remove_digits = str.maketrans('', '', digits)
    return text.translate(remove_digits)


def remove_stopwords(text):

    stopWords = set(stopwords.words('english'))
    content = word_tokenize(text)
    newText = ''

    for word in content:
        if word not in stopWords:
            newText = newText + word + ' '

    # https://stackoverflow.com/questions/12628958/remove-small-words-using-python
    # shortword = re.compile(r'\W*\b\w{1,2}\b')
    # newText = shortword.sub('', newText)
    
    return newText


def lemmatize(text):
    '''
    https://spacy.io/models/en
    Had to run this command in order to use english pipeline for CPU:
            python -m spacy download en_core_web_sm
    '''
    nlp = spacy.load('en_core_web_sm') # NOTE using md takes much longer
    doc = nlp(text)

    # Lemmatizing each token
    mytokens = [word.lemma_ if word.lemma_ != "-PRON-" else word.lower_ for word in doc]

    return ' '.join(mytokens)


def create_docterm_matrix():
    '''
    Perform text vectorizaion by reading scripts and converting into document-term dataframes

    creates:
    special_titles.txt, name of each special
    docterm_matrix.csv, result of TF-IDF vectorization
    '''

    # Measure wall time
    wall_start = time.time()
    
    # scripts to skip due to scraping errors
    skips = [55, 96, 97, 99, 203, 267, 268, 269, 271, 272, 274, 275, 286, 287, 336]

    # Text file where each line will be the next special of docterm matrix
    titles_file = open('special_titles.txt', 'w')

    # list of pruned transcipt strings, to be used in TFIDF
    preprocessed_transcripts = []

    # 372 total specials scraped
    num_scripts = 372

    for script_num in tqdm( range(num_scripts) , desc="Loading…" , ascii=False , ncols=75 ):
        
        # skip scripts that returned no data when scraping
        if script_num in skips:
            continue

        script_path = "scripts/script" + str(script_num)
        f = open(script_path, "r")
        title = f.readline()
        title = ' '.join(title.split('-'))

        # skip scripts that contain any of the following keywords in their title
        skip_tokens = ['snl', 'saturday', 'italiano', 'italiana', 'completa', 'speech', 'monologue', 'show']
        valid_special = True
        for s in skip_tokens:
            if s in title:
                valid_special = False
                break
        if not valid_special:
            continue

        content = f.read()

        # preprocessing tasks
        content = content.lower()
        content = remove_ints(content)
        content = remove_stopwords(content)
        content = lemmatize(content)

        preprocessed_transcripts.append(content)
        titles_file.write(title)

    # Text vectorization
    TFIDF = TfidfVectorizer()
    vec = TFIDF.fit_transform(preprocessed_transcripts)
    df = pd.DataFrame(vec.toarray(), columns=TFIDF.get_feature_names())

    # create 2 files for matrix and list of titles
    df.to_csv(f'docterm_matrix.csv')
    titles_file.close()

    print(f'Wall time: {round(time.time() - wall_start, 2)}s')


def create_query_vector(query):
    '''
    Read text and convert to a vector, just like above function

    returns: a dataframe containing TF-IDF vector for (string) query
    '''

    # preprocess tasks
    query = query.lower()
    query = remove_ints(query)
    query = remove_stopwords(query)
    query = lemmatize(query)

    # both methods garner identical results barring duplicate words in search query
    # VecTrans = TfidfVectorizer()
    VecTrans = CountVectorizer()
    vec = VecTrans.fit_transform([query])
    return pd.DataFrame(vec.toarray(), columns=VecTrans.get_feature_names())