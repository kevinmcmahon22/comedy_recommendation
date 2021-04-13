from sklearn.metrics.pairwise import cosine_similarity
from preprocess import create_keywords_vector
from tqdm import tqdm
import pandas as pd


def compute_cosine(keywords, n=10, num_specials=326):
    '''
    Compute cosine similarity between keywords and each special
    Print top n specials ranked by cosine similarity
    Use tqdm to print a progress bar in terminal
    
    TODO
    takes 2.5 secs/file, 2:15 for one query
        is there a faster way?

    integrate name of special into recs, maybe with a higher level of importance
        bag of words for similarity of title

    alternative to cosine similarity?
        TS-SS, however cost may be much higher

    compute cosine similarity as one large matrix calc?
        vectors to multiply would be very large - probs not worth it

    shorter documents tend to get precendence over longer ones?
        politically correct language is relatively short
    '''
    
    cos_sim = []
    key_df_origin = create_keywords_vector(keywords)
    f = open('special_titles.txt')

    print(key_df_origin)

    for i in tqdm( range(num_specials) , desc="Loading…" , ascii=False , ncols=75 ):

        # get title of next special
        special_name = f.readline()

        # get df for keywords and file
        key_df = key_df_origin.copy()
        file_df = pd.read_csv(f'vectors/vec{i}.csv')
        file_df = file_df.drop(['Unnamed: 0'], axis=1) # drop unecessary column created by file

        # Add headers of file_df to key_df with value of zero if they are not already in it
        for word in file_df.columns:
            if word not in key_df.columns:
                key_df[word] = 0

        # and vice versa
        for word in key_df.columns:
            if word not in file_df.columns:
                file_df[word] = 0

        # calculate cosine similarity, add to list with title
        sim = cosine_similarity(key_df, file_df)[0][0]
        cos_sim.append( (special_name.strip(), sim) )
    
    cos_sim.sort(key = lambda x: x[1], reverse=True)
    df = pd.DataFrame(cos_sim)
    print(df[:n])