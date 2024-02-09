import os 
import sys
from preprocess import readFiles

def querysplitter(text):
   sections = text.split('<num>') # diff sections of the query file

   for s in sections: # loop over each section (delimited by the tag)
      if s.strip() == '':
         continue
      
      num = s.split('\n')[0].strip() # strip based on the new line after the tag

      with open(f'{num}.txt', 'w') as f:
         f.write(s)

def main():
   querysplitter('./topics1-50.txt')





