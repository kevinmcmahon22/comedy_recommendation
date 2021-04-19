from scrape import create_transcript_files
from preprocess import create_docterm_matrix
from similarity import compute_similarity


def main():

    # comment when scraping done
    # create_transcript_files()

    # comment when vector files created
    # create_docterm_matrix()

    query = input()
    compute_similarity(query)

    '''
    TODO
    make faster by keeping docterm matrix on a server so it doesn't have to be opened every time

    cosine similarity for comedian name/title

    maybe compute cosine similarity across specials
        if user likes one special suggest others they may like based on cosine similarity
        
    SVD - primarilty for collaborative filtering
    '''


if __name__ == "__main__":
    main()