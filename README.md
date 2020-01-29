## Pipeline ##
Download Mediawiki dataset dumps -> Extract (title, text, id) from dataset dumps if text length (in chars) > 1000 -> Extract Only the first 1000 chars of the text
Save each article found as a json object in a new lines (.jsonl fomrat).

### Time improvements ###
Finally extracted files size currently 3.08GB (dump size total compressed ~17GB)


# Info for me #
Article amount before: 1177440 (text length > 200)
Article amount after: 1061896 (text length > 1000)
