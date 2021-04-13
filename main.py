from scrape import create_transcript_files
from preprocess import create_transcript_vectors
from similarity import compute_cosine


def main():

    # comment when scraping done
    # create_transcript_files()

    # comment when vector files created
    # create_transcript_vectors()

    keywords = input()
    compute_cosine(keywords, num_specials=20)    


if __name__ == "__main__":
    main()