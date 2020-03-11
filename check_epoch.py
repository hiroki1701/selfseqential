import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--itr', default=0, type=int, help='output h5 file')
args = parser.parse_args()

epoch = args.itr // 1564.5
print(epoch)