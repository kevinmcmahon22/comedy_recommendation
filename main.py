from scrape import create_transcript_files
from preprocess import create_docterm_matrix
from similarity import compute_similarity
import argparse

'''
Potential future improvements
    Store csv files on db so it doesn't take so long to load them
    Make website with UI
    Add user accounts in order to implement collaborative filtering
'''

# comment when scraping is complete
# create_transcript_files()

# comment when vector files have been
# create_docterm_matrix()

parser = argparse.ArgumentParser(description='Comedy RecSys')
parser.add_argument('-T', action='store_true', help='Add -T flag search by title instead of query')
args = parser.parse_args()

prompt = '\nEnter name of a special: ' if args.T else '\nEnter search query: '
query = input(prompt)

compute_similarity(query, args.T)
