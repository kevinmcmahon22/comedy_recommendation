from scrape_to_files import createTransciptFiles
from preprocess import to_vectors

def main():
    # comment this line if scraping already complete
    # createTransciptFiles()

    # convert files to docterm matrix
    to_vectors()


if __name__ == "__main__":
    main()