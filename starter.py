#!/usr/bin/python3

"""
    search.py

    MediaWiki API Demos
    Demo of `Search` module: Search for a text or title

    MIT License
"""

import os
import datetime
import requests

S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

SEARCHPAGE = "Nelson Mandela"

PARAMS = {
    "format": "json",
    "action": "query",
    "generator": "search",
    "gsrnamespace": 0,
    "gsrsearch": "Berlin",
    "gsrlimit": 10,
    "prop": "extracts",
    "pilimit": "max",
    "exintro": True,
    "explaintext": True,
    "exsentences": 1,
    "exlimit": "max"
}

R = S.get(url=URL, params=PARAMS)
DATA = R.json()

time_info = datetime.datetime.now().isoformat(timespec='minutes')
with open(os.getcwd() + '/example_results/' + time_info + '.json', 'w') as fp:
    import json
    json.dump(DATA, fp, ensure_ascii=False, indent=4)

# if DATA['query']['search'][0]['title'] == SEARCHPAGE:
#     print("Your search page '" + SEARCHPAGE + "' exists on English Wikipedia")
