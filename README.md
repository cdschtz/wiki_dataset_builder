# Wikipedia Dataset Builder #
This repo functions as a tool to build a dataset of training samples consisting of machine generated and human generated pairs.<br>
Output samples are saved as json objects in each line and would look like this:<br>

> ###### hum_[...]_.jsonl######
> {"label": "human", "id": "12", "title": "This is a title"}

> ###### mach_[...].jsonl ######
> {"label": "machine", "id": "12", "title": "This is a title", "text": "This is machine generated text"}



## Pipeline ##
Download Mediawiki dataset dumps -> Extract (title, text, id) from dataset dumps if text length (in chars) > 1000 -> Extract Only the first 1000 chars of the text <br>
Save each article found as a json object in a new lines (.jsonl fomrat).

### Time improvements ###
Finally extracted files size currently 3.08GB (dump size total compressed ~17GB)


#### Info for me ####
Article amount before: 1177440 (text length > 200) <br>
Article amount after: 1061896 (text length > 1000)

from google.colab import drive
drive.mount('/content/drive')
