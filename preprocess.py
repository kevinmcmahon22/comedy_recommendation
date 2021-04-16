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
# nltk.download('punkt')
nltk.download('stopwords')


def remove_ints(text):
    '''
    remove any tokens that are numbers/integers

    https://stackoverflow.com/questions/12851791/removing-numbers-from-string
    '''
    # This version doesn't work in Python3
    # return text.translate(None, digits)
    remove_digits = str.maketrans('', '', digits)
    return text.translate(remove_digits)


def remove_stopwords(text):
    text = text.lower()
    # https://stackoverflow.com/questions/12628958/remove-small-words-using-python
    shortword = re.compile(r'\W*\b\w{1,2}\b')

    stopWords = set(stopwords.words('english'))
    content = word_tokenize(text)
    newText = ''

    for word in content:
        if word not in stopWords:
            newText = newText + word + ' '
    
    return shortword.sub('', newText)


def lemmatize(text):
    '''
    Had to run this command in order to use english pipeline for CPU:

    python -m spacy download en_core_web_md

    https://spacy.io/models/en, md stands for medium
    '''
    nlp = spacy.load('en_core_web_md')
    doc = nlp(text)

    # Lemmatizing each token
    mytokens = [word.lemma_ if word.lemma_ != "-PRON-" else word.lower_ for word in doc]

    return ' '.join(mytokens)


def create_transcript_vectors():
    '''
    Reading scripts and converting into document-term dataframes

    creates:
    special_titles.txt, containing the name of each special
    vectors dir, each vec file contains a csv of tf-idf vectors
    returns: list of 326 pandas df containing vectorized doc terms
    '''
    
    # path = "/content/drive/My Drive/CSE881 Group Project/scripts/script"

    # Measure wall and cpu execution time
    wall_start = time.time()
    
    # scripts to skip, scraping errors
    skips = [55, 96, 97, 99, 203, 267, 268, 269, 271, 272, 274, 275, 286, 287, 336]

    # Text file where each line will be the title of special
    # Line number corresponds to vec_num used to number TF-IDF csv files
    titles_file = open('special_titles.txt', 'w')
    vec_num = 0

    # 372 total specials scraped
    for script_num in tqdm( range(372) , desc="Loading…" , ascii=False , ncols=75 ):
        
        # skip files that returned no data when scraping
        if script_num in skips:
            continue

        # Google Drive/Collab
        # final_path = path + str(i)
        # f = open(final_path, "r")
        # title = f.readline()

        script_path = "scripts/script" + str(script_num)
        f = open(script_path, "r")
        title = f.readline()
        title = ' '.join(title.split('-'))

        # don't process scripts that contain any of the following keywords in their title
        skip_tokens = ['snl', 'saturday', 'italiano', 'speech', 'monologue', 'show']
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

        # Count vectorizer
        # vector = CountVectorizer()
        # arr = vector.fit_transform([content])
        # df = pd.DataFrame(arr.toarray(), columns=vector.get_feature_names())

        # TF-IDF transformation
        TFIDF = TfidfVectorizer()
        vec = TFIDF.fit_transform([content])
        df = pd.DataFrame(vec.toarray(), columns=TFIDF.get_feature_names())

        # Write title and df to files
        titles_file.write(title)
        df.to_csv(f'vectors/vec{vec_num}.csv')
        vec_num += 1

    # flush buffer, actually write to file
    titles_file.close()

    # Wall time: 3768.74403s
    print(f'Wall time: {round(time.time() - wall_start, 5)}s')


def create_keywords_vector(keywords):
    '''
    Read text and convert to a vector, just like above function

    returns: a dataframe containing TF-IDF vector for (string) keywords
    '''

    # preprocess tasks
    keywords = keywords.lower()
    keywords = remove_ints(keywords)        # NOTE: remove year numbers in title
    keywords = remove_stopwords(keywords)
    # keywords = lemmatize(keywords)        # NOTE: comment out since by far most expensive preprocess task

    # barring duplicate words in search query, both methods lead to identical results
    # VecTrans = TfidfVectorizer()
    VecTrans = CountVectorizer()
    vec = VecTrans.fit_transform([keywords])
    return pd.DataFrame(vec.toarray(), columns=VecTrans.get_feature_names())