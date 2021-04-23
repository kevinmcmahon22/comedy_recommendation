from scrape import create_transcript_files
from preprocess import create_docterm_matrix
from similarity import compute_similarity


'''
Future improvements
    Store csv files on db so it doesn't take so long to load them
    Make website with UI
    Add user accounts in order to implement collaborative filtering
'''

# comment when scraping done
# create_transcript_files()

# comment when vector files created
# create_docterm_matrix()

queryType = input("Enter Recommendation Type (1 for content based or 2 for title based): ")
query = input("Enter Search Query: ")
compute_similarity(query, queryType)
