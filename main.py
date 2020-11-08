from __future__ import print_function
import argparse
from data_prepossing import load_dataset
from train import *

def argument():
  
  parser = argparse.ArgumentParser(description='Road Damage Classification...')
  parser.add_argument('-epoch', type=int, default=20, help='')
  parser.add_argument('-path', type=str, default="data/", help='path to image')
  args = parser.parse_args()

  return args


def main(args):
  
  x, y = load_dataset()
  
  num_classes = 5
  model = load_model(num_classes)
  preview_model(model)
	
  train_model(x,y,model,epoch=args.epoch)


if __name__ == '__main__':
  
  args = argument()
  print(args.epoch)
  print(args.path)
  
  main(args)