from torchtext import data
from torchtext import datasets
import torch
import os

import csv
import sys
csv.field_size_limit(sys.maxsize)

def make_rt_gender(batch_size, base_path, train_file, valid_file, test_file, device=-1, vectors=None, topics=False):
  TEXT = data.Field(include_lengths=True, lower=True)
  LABEL = data.LabelField()
  INDEX = data.Field(sequential=False, use_vocab=False, batch_first=True)

  if topics:
    TOPICS = data.Field(sequential=True, use_vocab=False, preprocessing=data.Pipeline(lambda x:float(x)), tensor_type=torch.cuda.FloatTensor, batch_first=True)
    train = data.TabularDataset(path=os.path.join(base_path,train_file), format="tsv", fields=[('index', INDEX), ('text',TEXT), ('label', LABEL), ('topics', TOPICS)])
  else:
    train = data.TabularDataset(path=os.path.join(base_path,train_file), format="tsv", fields=[('index', INDEX), ('text',TEXT), ('label', LABEL)])

  TEXT.build_vocab(train, vectors=vectors, max_size=30000)
  LABEL.build_vocab(train)
  print (LABEL.vocab.stoi)

  val = data.TabularDataset(path=os.path.join(base_path,valid_file), format="tsv", fields=[('index', INDEX), ('text',TEXT), ('label', LABEL)])
  test = data.TabularDataset(path=os.path.join(base_path,test_file), format="tsv", fields=[('index', INDEX), ('text',TEXT), ('label', LABEL)])

  train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), batch_sizes=(batch_size, 256, 256), device=device, repeat=False, sort_key=lambda x: len(x.text))

  if topics:
    return (train_iter, val_iter, test_iter), TEXT, LABEL, TOPICS, INDEX
  else:
    return (train_iter, val_iter, test_iter), TEXT, LABEL, INDEX

# this doesn't expect a test data set or topics
def make_rt_gender_op_posts(batch_size, base_path, train_file, valid_file, test_file = None, device=-1, vectors=None):
  TEXT = data.Field(include_lengths=True, lower=True)
  LABEL = data.LabelField()
  INDEX = data.Field(sequential=False, use_vocab=False, batch_first=True)
  train = data.TabularDataset(path=os.path.join(base_path, train_file), format="tsv", fields=[('index', INDEX), ('text',TEXT), ('label', LABEL)])
  val = data.TabularDataset(path=os.path.join(base_path, valid_file), format="tsv", fields=[('index', INDEX), ('text',TEXT), ('label', LABEL)])

  if test_file is not None and test_file != "":
    test = data.TabularDataset(path=os.path.join(base_path, test_file), format="tsv", fields=[('index', INDEX), ('text',TEXT), ('label', LABEL)])

  TEXT.build_vocab(train, vectors=vectors, max_size=30000)
  LABEL.build_vocab(train)
  print (LABEL.vocab.stoi)
  if test_file is not None and test_file != "":
    train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), batch_sizes=(batch_size, 256, 256), device=device, repeat=False, sort_key=lambda x: len(x.text))
    return (train_iter, val_iter, test_iter), TEXT, LABEL, INDEX

  train_iter, val_iter = data.BucketIterator.splits((train, val), batch_sizes=(batch_size, 256), device=device, repeat=False, sort_key=lambda x: len(x.text))
  return (train_iter, val_iter), TEXT, LABEL, INDEX
