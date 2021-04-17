import os
import torch
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np
import pdb
import operator
import random
import spacy

import torch.cuda as cuda
USE_CUDA = cuda.is_available()

# import the config file
import json
config_file = 'config.json'
config = json.load(open(config_file, 'r'))

data_dir = config['data_dir']
image_data_dir = config['image_data_dir']
train_data_file = config['train_data_file']
test_data_file = config['test_data_file']
val_data_file = config['val_data_file']
image_feat_data_dir = config['image_feat_data_dir']

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    try:
#         (image_file, image_vec, abstract, paragraph, noun_pos, ner_pos) = zip(*data)
        (image_file, image_vec, abstract, paragraph) = zip(*data)
#         (image_file, image_vec, abstract, paragraph, noun_pos) = zip(*data)
        #print("Data is iterable.")
        #print(len(data))
    except:
        print("Data is NOT iterable!!!!")
        #print(len(data))
        return None
    
    batch_images = torch.stack(image_vec, 0).squeeze(1)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    # Captions in our dataset are not explicit descriptions of the image content. Hence we call them 'abstract captions'.
    abstract_lengths = [len(abs) for abs in abstract]
    batch_abstract = torch.zeros(len(abstract), config['max_cap_len']).long()
    for i, abs in enumerate(abstract):
        end = abstract_lengths[i]
        batch_abstract[i, :end] = abs[:end]

    # Paragraph
    par_lengths = [len(par) for par in paragraph]
    batch_paragraph = torch.zeros(len(paragraph), config['max_par_len']).long()

    for i, par in enumerate(paragraph):
        end = par_lengths[i]
        batch_paragraph[i, :end] = par[:end]

    return (image_file, batch_images, batch_abstract, batch_paragraph)
    
# Create vocabulary
class Vocabulary():
    def __init__(self):
        self.w_to_idx = {'<pad>': 0, '<sos>': 1, '<UNK>': 3, '<eos>': 2}
        self.idx_to_w = {0: '<pad>', 1: '<sos>', 3: '<UNK>', 2: '<eos>'}
        self.vocabulary = []
        self.idx_to_personality = dict()
        self.w_count = dict()
        self.prepro_data = dict()

    def tokenize(self, s):
        s = s.lower()
        s = s.strip('\n').strip('\r')
        s = s.replace('"', ' "')
        for w in s.split(' '):
            if w not in self.w_count:
                self.w_count[w] = 1
            else:
                self.w_count[w] += 1

    def build_vocab(self, data_file):
        with open(data_file, 'r') as f:
            data = json.load(f, encoding='utf8')
            for idx, doc in enumerate(data):
                abs_cap = doc['caption']
                self.tokenize(abs_cap)

                paragraph = doc['text']
                self.tokenize(paragraph)

        # if we need to choose top k words
        sorted_x = sorted(self.w_count.items(), key=operator.itemgetter(1), 
                          reverse=True)
        count = 0
        for w in sorted_x:
            if count < config['vocab_size']: # To truncate vocab uncomment this line and indent the following
                if w[0] not in self.w_to_idx:
                    idx = len(self.w_to_idx)
                    self.w_to_idx[w[0]] = idx
                    self.idx_to_w[idx] = w[0]
            count += 1
            # else:
            #     break
        
        self.vocabulary = list(self.w_to_idx.keys())
        
        self.prepro_data['vocab'] = self.vocabulary
        self.prepro_data['w2i'] = self.w_to_idx
        self.prepro_data['i2w'] = self.idx_to_w
        self.prepro_data['word_count'] = self.w_count
        print("Vocabulary size = {}".format(len(self.vocabulary)))
        return self.prepro_data

class ProcessSentence():
    def __init__(self, v):
        self.w_to_idx = v['w2i']
        self.max_cap_len = config['max_cap_len']
        self.max_par_len = config["max_par_len"]
        # Load English tokenizer, tagger, parser, NER and word vectors
        self.nlp = spacy.load("en_core_web_sm")

    def vectorize_caption(self, s):
        vec = []
        s = s.lower()
        s = s.strip('\n').strip('\r')
        s = s.replace('"', ' "')
        words = s.split(' ')
        vec.append(self.w_to_idx['<sos>'])
        for w in words:
            if w in self.w_to_idx:
                idx = self.w_to_idx[w]
                vec.append(idx)
            else: 
                vec.append(self.w_to_idx['<UNK>'])
        vec.append(self.w_to_idx['<eos>'])
        length = len(vec)

        if len(vec) > self.max_cap_len:
            vec = vec[0:self.max_cap_len]
            length = self.max_cap_len
        return vec, length

    def vectorize_paragraph(self, s):
        vec = []
        noun_pos_vec = [] # if word is a noun 1, else 0
        ner_pos_vec = [] # if word is a named entity 1, else 0
        s = s.lower()
        s = s.strip('\n').strip('\r')
        s = s.replace('"', ' "')
        words = s.split(' ')
        
        s_spacy = self.nlp(s) # NLP processed paragraph
        s_nouns = [token.text for token in s_spacy if token.pos_ == "NOUN" or token.pos_ == "PROPN"]
        s_ners = [entity.text for entity in s_spacy.ents] # named entities
        
        
        vec.append(self.w_to_idx['<sos>'])
        noun_pos_vec.append(0)
        ner_pos_vec.append(0)
        
        for w in words:
            if w in self.w_to_idx:
                idx = self.w_to_idx[w]
                vec.append(idx)
                if w in s_nouns or w in s_ners: 
                    noun_pos_vec.append(1) # if word is a NE, it is also a noun
                    if w in s_ners:
                        ner_pos_vec.append(1)
                    else:
                        ner_pos_vec.append(0)
                else:
                    noun_pos_vec.append(0)
                    ner_pos_vec.append(0)
                                  
            else:
                vec.append(self.w_to_idx['<UNK>'])
                noun_pos_vec.append(0)
                ner_pos_vec.append(0)
                
        vec.append(self.w_to_idx['<eos>'])
        noun_pos_vec.append(0)
        ner_pos_vec.append(0)
        
        length = len(vec)

        if len(vec) > self.max_par_len:
            vec = vec[0:self.max_par_len]
            noun_pos_vec = noun_pos_vec[0:self.max_par_len]
            ner_pos_vec = ner_pos_vec[0:self.max_par_len]
            length = self.max_par_len
            
#         return vec, length, noun_pos_vec, ner_pos_vec
        return vec, length


class Dataset(Dataset):
    def __init__(self, data_file, vocab, tokenizer=None):
        with open(data_file, 'r') as f:
            self.data = json.load(f, encoding='utf8')
        self.v = vocab
        self.preprocess = ProcessSentence(self.v)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #****** Generates one sample of data *******
        try:
            # get image features
            img_id = self.data[idx]['image_hash']
            image_file = os.path.join(image_feat_data_dir, img_id + '.npy')
            image_vec = torch.from_numpy(np.load(image_file))
#            image_vec = torch.zeros(1, 2048) # for text-summarization, we reinitialise the image vector as  zeros
            # print("image vec >>>", image_vec)
    
            # get abstract caption
            abs_cap = self.data[idx]['caption']
            if self.tokenizer is not None:
                tokens = self.tokenizer.tokenize(abs_cap)
                abstract = self.tokenizer.convert_tokens_to_ids(tokens)
                abstract = [self.tokenizer.bos_token_id] + abstract + [self.tokenizer.eos_token_id]
                if len(abstract) > config['max_cap_len']:
                  abstract = abstract[0:config['max_cap_len']]
                length = len(abstract)
            else:
                abstract, length = self.preprocess.vectorize_caption(abs_cap)
            abstract = torch.LongTensor(abstract)
            
            # get paragraph
            if self.tokenizer is not None:
                tokens = self.tokenizer.tokenize(self.data[idx]['text'])
                paragraph = self.tokenizer.convert_tokens_to_ids(tokens)
                paragraph = [self.tokenizer.bos_token_id] + paragraph + [self.tokenizer.eos_token_id]
                if len(paragraph) > config['max_par_len']:
                  paragraph = paragraph[0:config['max_par_len']]
                par_len = len(paragraph)
            else:
                paragraph, par_len = self.preprocess.vectorize_paragraph(self.data[idx]['text'])

            # get paragraph and noun/ner position vectors (if word is a noun/ner, 1, else 0)
#             paragraph, par_len, noun_pos_vec, ner_pos_vec = self.preprocess.vectorize_paragraph(self.data[idx]['text'])
            # get paragraph for simple captioning (no paragraph)
#            paragraph, par_len = self.preprocess.vectorize_paragraph("")
            
            paragraph = torch.LongTensor(paragraph)
#             noun_pos_vec = torch.LongTensor(noun_pos_vec)
#             ner_pos_vec = torch.LongTensor(ner_pos_vec)
            
            # image_file needed for qualitative assessment during inference
            return (image_file, image_vec, abstract, paragraph) # paragraph; simple attention
#             return (image_file, image_vec, abstract, paragraph, noun_pos_vec, ner_pos_vec) #paragraph, attention on noun and ner

        except:
            #print(self.data[idx]['image_hash'], self.data[idx]['text'])
            print("Error generating data sample.")