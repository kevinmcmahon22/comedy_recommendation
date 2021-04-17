from sklearn.metrics.pairwise import cosine_similarity
from preprocess import create_keywords_vector
from tqdm import tqdm
import pandas as pd


def compute_cosine(keywords, n=10, num_specials=326):
    '''
    Compute cosine similarity between keywords and each special
    Print top n specials ranked by cosine similarity
    Use tqdm to print a progress bar in terminal
    
    NOTE currently takes 2m 15s, only 12% of CPU dedicated for processing
    '''
    
    cos_sim = []

    f = open('special_titles.txt')

    df_docterm = pd.read_csv('docterm_matrix.csv')
    df_docterm = df_docterm.drop(['Unnamed: 0'], axis=1)

    df_query_temp = create_keywords_vector(keywords)

    # Remove words from query if not found in document corpus
    df_query = pd.DataFrame()
    for word in df_query_temp.columns:
        if word in df_docterm.columns:
            df_query[word] = df_query_temp[word]

    print(df_query)


    for i in tqdm( range(num_specials) , desc="Loading…" , ascii=False , ncols=75 ):

        # get title and df for next special
        special_name = f.readline()
        series_special = df_docterm.loc[i, :]
        df_special = series_special.to_frame().swapaxes("index", "columns")

        # calculate cosine similarity, add to list with title
        sim = cosine_similarity(df_query, df_special)[0][0]
        cos_sim.append( (special_name.strip(), sim) )
    
    # Show results
    cos_sim.sort(key = lambda x: x[1], reverse=True)
    df = pd.DataFrame(cos_sim)
    print(df[:n])