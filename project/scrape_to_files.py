from bs4 import BeautifulSoup
import requests
import re


def getURLs():
    '''
    Scrape landing page for URL for each comedy special's URL

    returns: list
    '''
    page = requests.get("https://scrapsfromtheloft.com/stand-up-comedy-scripts/")
    soup = BeautifulSoup(page.content, 'html.parser')
    transcript_urls = [t['href'] for t in soup.find_all('a')[67:440]]
    return transcript_urls

def getOneTranscipt(url):
    '''
    Scrape url for contents of comedy special

    returns: string
    '''

    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    text = soup.find_all('p')
    script_lines = []
    for line in text[:-10]:
        script_lines.append(line.get_text())

    script = ' '.join(script_lines)

    # replace non-alphanumeric characters with a space
    script = re.sub("[^0-9a-zA-Z]+", " ", script)

    return script

def create_transcript_files():
    '''
    Add unique identifying tokens and transcript to a text file

    Stored in local scripts directory

    returns: None
    '''

    # GET list of urls for each comedy transcript
    # transcript_urls = getURLs()

    # Create file containing list of urls
    # with open('urls.txt','w') as f:
    #     for t in transcript_urls:
    #         f.write(t + '\n')

    f = open('urls.txt', 'r')
    for i, url in enumerate(f.readlines()):

        # Retrieve specifying tokens from url
        url_tokens = url[30:-2].strip().split('/')[-1].split('-')

        # remove meaningless tokens, get final list of tokens that describle stand-up special
        final_description = []
        useless = ['full', 'transcript']
        for token in url_tokens:
            if token not in useless:
                final_description.append(token)
        
        # GET transcript as a string
        script_string = getOneTranscipt(url)
        
        # create a file containing one comedy transcript
        new_f = open(f'scripts/script{i}', 'w')
        new_f.write('-'.join(final_description) + '\n')
        new_f.write(script_string)
        new_f.close()
