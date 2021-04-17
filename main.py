from scrape import create_transcript_files
from preprocess import create_docterm_matrix
from similarity import compute_cosine


def main():

    # comment when scraping done
    # create_transcript_files()

    # comment when vector files created
    # create_docterm_matrix()

    query = input()
    compute_cosine(query)

    # SVD - primarilty for collaborative filtering



if __name__ == "__main__":
    main()