#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import time
import os
import urllib
import urllib.request as urllib2
import wget
import zipfile
#from StringIO import StringIO

from progressbar import *
#from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
#    AdaptiveETA, FileTransferSpeed, FormatLabel, Percentage, \
#    ProgressBar, ReverseBar, RotatingMarker, \
#    SimpleProgress, Timer


def download_file_by_url(url, path_to_file, toPrint=False, overwrite=False):
    directory = path_to_file
    if not os.path.exists(directory):
        os.makedirs(directory)
    site = urllib2.urlopen(url)
    meta = site.info()
    file_name = url.split('/')[-1]
    file_nameBase = file_name.split('.')[0]

    if os.path.isfile(os.path.join(directory, file_nameBase, file_name)) and overwrite==False:
        if toPrint == True:
            print(os.path.join(directory, file_nameBase, file_name), " already present")
        return (file_name, os.path.join(directory, file_nameBase))

    if not os.path.exists(os.path.join(directory, file_nameBase)):
        os.makedirs(os.path.join(directory, file_nameBase))

    f = open(os.path.join(directory, file_nameBase, file_name), 'wb')
    try:
        file_size = int(site.getheader("Content-Length"))
    except IndexError:
        response = urllib2.urlopen(url)
        html = response.read()
        f.write(html)
        f.close()
        return (file_name, os.path.join(directory, file_nameBase))
    u = urllib2.urlopen(url)
    #f = open(file_name, 'wb')


    print("Downloading: %s Bytes: %s" % (file_name, file_size))
    widgets = [Bar('>'), ' ', ETA(), ' ', ReverseBar('<')]
    pbar = ProgressBar(widgets=widgets, maxval=file_size).start()

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break
        file_size_dl += len(buffer)
        f.write(buffer)
        #status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        #status = status + chr(8)*(len(status)+1)
        #print status,
        pbar.update(file_size_dl)
    pbar.finish()
    f.close()
    return (file_name, os.path.join(directory, file_nameBase))

def unzip_file(filename, path_to_file, toPrint=False):
    if os.path.isfile(os.path.join(path_to_file, filename)):
        with zipfile.ZipFile(os.path.join(path_to_file, filename)) as zf:
            zf.extractall(path=path_to_file)
            

def main():
    directory = "data/"
    links = ["https://s3.amazonaws.com/babs-open-data/babs_open_data_year_1.zip",
             "https://s3.amazonaws.com/babs-open-data/babs_open_data_year_2.zip"]
    for url in links:
        #url = "http://www.mapcruzin.com/download-shapefile/norway-railways-shape.zip"
        #file_name = url.split('/')[-1]
        (filename, path_to_file) = download_file_by_url(url, directory)
        unzip_file(filename, path_to_file)
    return 0

if __name__ == '__main__':
    main()
