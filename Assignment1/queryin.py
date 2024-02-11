import os 
import sys
from preprocess import readFiles
import json

def querysplitter(text):
   sections = text.split('<num>') # diff sections of the query file

   for s in sections: # loop over each section (delimited by the tag)
      if s.strip() == '':
         continue
      
      num = s.split('\n')[0].strip() # section/query number

      f = open('./queries/' + num + '.txt', 'w')
      f.write(s)
      f.close()

path = './topics1-50.txt'
with open(path, 'r') as f:
   text = f.read()
querysplitter(text)





