from sklearn.metrics.pairwise import cosine_similarity
from preprocess import create_query_vector
from tqdm import tqdm
import pandas as pd
import numpy as np
import time


def compute_similarity(query, n=10, metric='cos'):
    '''
    Compute cosine similarity between search query and each special
    Print top n specials ranked by cosine similarity
    '''

    wall_start = time.time()
    f = open('special_titles.txt')
    df_docterm = pd.read_csv('docterm_matrix.csv')
    df_docterm = df_docterm.drop(['Unnamed: 0'], axis=1)
    print(f'Opened files in {round(time.time() - wall_start, 2)}s\n')

    # Create query df of proper size, remove words in query not present in document corpus
    wall_start = time.time()
    df_query = pd.DataFrame( np.zeros((1,len(df_docterm.columns)),dtype=int) , columns=df_docterm.columns )
    df_query_temp = create_query_vector(query)
    keywords_used = []
    for word in df_query_temp:
        if word in df_query.columns:
            df_query[word] = df_query_temp[word]
            keywords_used.append(word)
    print(f'Created dataframe for query in {round(time.time() - wall_start, 2)}s\n')

    cos_sim = []

    titles = f.readlines()
    it = iter(titles)

    # for i in tqdm( range(num_specials) , desc="Loading…" , ascii=False , ncols=75 ): # NOTE runs so fast, no loner needed
    for i in range(len(titles)):

        # get title and df for next special
        special_name = next(it)
        series_special = df_docterm.loc[i, :]
        df_special = series_special.to_frame().swapaxes("index", "columns")

        # NOTE potential to add more pairwise metrics here
        if metric == 'cos':
            sim = cosine_similarity(df_query, df_special)[0][0]
        cos_sim.append( (special_name.strip(), sim) )
    
    # Output results
    cos_sim.sort(key = lambda x: x[1], reverse=True)
    df = pd.DataFrame(cos_sim, columns=['Name of special', 'Score'])
    print(f'TOP {n} SPECIALS BASED ON QUERY "{query}"\n')
    print(f'\tTokens used: ', end='')
    for i in keywords_used:
        print(f'{i} ', end='')
    print('\n')
    print(df[:n])