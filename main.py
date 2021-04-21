from scrape import create_transcript_files
from preprocess import create_docterm_matrix
from similarity import compute_similarity


def main():

    # comment when scraping done
    # create_transcript_files()

    # comment when vector files created
    # create_docterm_matrix()

    queryType = input("Enter Recommendation Type (1 for content based or 2 for title based): ")
    query = input("Enter Search Query: ")
    compute_similarity(query, queryType)

    '''
    TODO
    make faster by keeping docterm matrix on a server so it doesn't have to be opened every time

        
    SVD - primarilty for collaborative filtering
    '''


if __name__ == "__main__":
    main()