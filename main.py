from scrape import create_transcript_files
from preprocess import create_transcript_vectors, create_keywords_vector

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm


def main():

    # comment when scraping done
    # create_transcript_files()

    # comment when vector files created
    # create_transcript_vectors()


    # 
    # PUT UI here
    # 

    
    ''' TODO
    takes 2.5 secs/file, faster way?
    integrate name of special into recs, maybe with a higher level of importance
    alternative to cosine similarity?

    compute cosine similarity as one large matrix calc?
        vectors to multiply would be very large - probs not worth it
    '''


    f = open('special_titles.txt')

    # list of cosine similarity between each special and user input
    cos_sim = []

    # get user input and create a dataframe
    keywords = input()
    key_df_origin = create_keywords_vector(keywords)

    # number of specials we want to use in calculation, default 326
    num_specials = 10

    # use tqdm module to print a progress bar
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

    print(df[:10])


if __name__ == "__main__":
    main()