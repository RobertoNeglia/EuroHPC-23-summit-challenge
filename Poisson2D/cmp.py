#!/usr/bin/env python
import pandas as pd
import sys

if(len(sys.argv) > 3):
    print('Too many arguments')
    print('Correct usage: python cmp.py file1.csv file2.csv')
    exit()
if(len(sys.argv)<3):
    print('Too few arguments')
    print('Correct usage: python cmp.py file1.csv file2.csv')
    exit()
sol1 = pd.read_csv(sys.argv[1], sep=",")
sol2 = pd.read_csv(sys.argv[2], sep=",")

if(all(sol1.v == sol2.v)):
    print('THE TWO SOLUTIONS ARE THE SAME!')
else:
    print('NOOOOOOOO THE TWO SOLUTIONS ARE NOT THE SAME!')
