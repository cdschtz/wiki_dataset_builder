import requests

# Parsing HTML
from bs4 import BeautifulSoup

# File system management
import os


import bz2
import subprocess

file_path = '/Users/christopher/Downloads/enwiki-20200101-pages-articles2.xml-p30304p88444.bz2'
lines = []
for i, line in enumerate(subprocess.Popen(['bzcat'], stdin = open(file_path), stdout = subprocess.PIPE).stdout):
    lines.append(line)
    if i > 5e5:
        break


import xml.sax

class WikiXmlHandler(xml.sax.handler.ContentHandler):
    """Content handler for Wiki XML data using SAX"""
    def __init__(self):
        xml.sax.handler.ContentHandler.__init__(self)
        self._buffer = None
        self._values = {}
        self._current_tag = None
        self._pages = []

    def characters(self, content):
        """Characters between opening and closing tags"""
        if self._current_tag:
            self._buffer.append(content)

    def startElement(self, name, attrs):
        """Opening tag of element"""
        if name in ('title', 'id', 'text', 'timestamp'):
            self._current_tag = name
            self._buffer = []

    def endElement(self, name):
        """Closing tag of element"""
        if name == self._current_tag:
            self._values[name] = ' '.join(self._buffer)

        if name == 'page':
            self._pages.append((self._values['title'], self._values['text'], self._values['id']))



# Content handler for Wiki XML
handler = WikiXmlHandler()

# Parsing object
parser = xml.sax.make_parser()
parser.setContentHandler(handler)


for l in lines[0:1000]:
    parser.feed(l)