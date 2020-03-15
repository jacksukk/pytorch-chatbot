import json
import random
import re
import sys
import urllib.parse

import bs4
import requests

def get_response(query):

    query = re.sub(r'\s+', ' ', query).strip()
    query = urllib.parse.quote(query)

    # print('Retrieving response.')
    url = 'https://www.pandorabots.com/mitsuku/'
    session = requests.Session()
    session.headers.update({
        'User-Agent': ' '.join(('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_2)',
                                'AppleWebKit/537.36 (KHTML, like Gecko)',
                                'Chrome/72.0.3626.119 Safari/537.36')),
        'Referer': url
    })

    main_page = session.get(url).text
    main_soup = bs4.BeautifulSoup(main_page, 'lxml')
    botkey = re.search(r'PB_BOTKEY: "(.*)"', main_page).groups()[0]

# Mitsuku does not check your client_name really carefully (smirk) as
# long as the length is 13.
    client_name = str(random.randint(1337, 9696913371337)).rjust(13, '0')
    response_raw = session.post(
        f'https://miapi.pandorabots.com/talk?'
        f'botkey={botkey}&'
        f'input={query}&'
        f'client_name={client_name}&'
        f'sessionid=null&'
        f'channel=6').text
    try:
        response_json = json.loads(response_raw)
    except json.JSONDecodeError:
        # Sometimes Mitsuku sends stuff like pictures.
        response_json = {'responses': ["<i can't understand it, bro>"]}
# If the query is too long, Mitsuku answers in several lines.  Here
# I'll just put it into paragraphs.
    response = '\n\n'.join(response_json['responses'])
    # print(response)

    return response


if __name__ == '__main__':
    while True:
        sent = input()
        print(get_response(sent))
