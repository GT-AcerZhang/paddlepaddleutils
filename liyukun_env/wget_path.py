import os
import sys
import six
import argparse
import subprocess
import commands
import urllib2
import re

from HTMLParser import HTMLParser


def parse_args():
    parser = argparse.ArgumentParser("download model.")
    parser.add_argument(
        '--job_id',
        type=str,
        default=None)
    parser.add_argument(
        '--queue',
        type=str,
        default=None)
    parser.add_argument(
        '--start_steps',
        type=int,
        default=5000)
    parser.add_argument(
        '--skip_steps',
        type=int,
        default=5000)
    parser.add_argument(
        '--end_steps',
        type=int,
        default=10000)
    
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')
    

def scan_link(url):
    f = urllib2.urlopen(url)
    data = f.read()
    
    class MyHTMLParser(HTMLParser):
        def __init__(self):
            HTMLParser.__init__(self)
            self.models = []
        
    	def handle_data(self, data):
            if re.match('step', data.strip()):
                self.models.append(data.strip())
    
    parser = MyHTMLParser()
    models = parser.feed(data)
    return parser.models


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    get_host_name = 'showjob -j ' + args.job_id \
            + ' -p ' + args.queue \
            + ' | grep NodeList | tail -1 | cut -f2 -d"="' \
            + ' | cut -f1 -d","'
    retcode, host_name = commands.getstatusoutput(get_host_name)
    if retcode != 0:
        print(host_name)
    
    url_link = 'http://' + host_name + \
            ':8880/look/overview/dir?id=' + args.job_id + \
            '&file=/job-' + args.job_id + '/output'
    
    model_list = scan_link(url_link)

    for model in model_list:
        wget_path = 'http://' + host_name + ':8880/downloadfile/job-' \
                + args.job_id + '/output/' + model
        download_model = 'wget ' + wget_path
        retcode, ret = commands.getstatusoutput(download_model)
        print(ret)
        
        tar_model = 'tar -xvf ' + model
        retcode, ret = commands.getstatusoutput(tar_model)
        print(ret)
