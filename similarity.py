from sklearn.metrics.pairwise import cosine_similarity
from preprocess import create_query_vector
from tqdm import tqdm
import pandas as pd
import numpy as np
import time


def compute_cross_special_similarity(df_docterm, titles):
    '''
    if user likes one special suggest others they may like based on cosine similarity
    creates cosines.csv file, contains cosine similarity between every pair of specials in corpus (n=326)
    '''

    data = df_docterm.to_numpy()
    N = len(df_docterm)
    cosines = np.zeros((N, N))

    for row in tqdm( range(N) , desc="Creating cosines.csv…" , ascii=False , ncols=75 ):
        for rows in range(N):
            if row != rows:
                cosines[row, rows] = cosine_similarity(data[row, :].reshape(1, -1), data[rows, :].reshape(1, -1))[0][0]
            else:
                cosines[row, rows] = 1
    
    # Save pre-computed cosine similarities to cosines.csv 
    cos_df = pd.DataFrame(data=cosines, columns=titles)
    # cos_df.to_csv (r'/content/drive/My Drive/CSE881 Group Project/cosines.csv', index = False, header=True)
    cos_df.to_csv('cosines.csv', index = False, header=True)


def compute_similarity(query, byTitle, n=10, metric='cos'):
    '''
    Compute cosine similarity between search query and each special
    Print top n specials ranked by cosine similarity

    byTitle - bool, True if query is a title, false if it is a set of keywords to search for
    '''
    wall_start = time.time()
    f = open('special_titles.txt')
    titles = [x.strip() for x in f.readlines()]
    # Exit program if an invalid title is entered
    if query not in titles and byTitle:
        print('\n\tError: Please entery a valid special name\n')
        return 1

    df_docterm = pd.read_csv('docterm_matrix.csv')
    df_docterm = df_docterm.drop(['Unnamed: 0'], axis=1)
    print(f'\nOpened files in {round(time.time() - wall_start, 2)}s\n')
    

    if not byTitle:
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


    else:
        # Only need to run next line once, comment out
        # compute_cross_special_similarity(df_docterm, titles)

        # cosines = pd.read_csv('/content/drive/My Drive/CSE881 Group Project/cosines.csv')
        cosines_df = pd.read_csv('cosines.csv')

        # exclude top 1 result, special will have similarity of 1 with itself
        order = np.argsort(-cosines_df[query].values)[1:n+1]

        order_list = []
        for i in order:
            order_list.append( (titles[i], cosines_df.iloc[i, cosines_df.columns.get_loc(query)]) )

        order_list.sort(key = lambda x: x[1], reverse=True)
        df = pd.DataFrame(order_list, columns=['Name of special', 'Score'])
        print(f'TOP {n} SPECIALS SIMILAR TO "{query}"\n')
        print(df)