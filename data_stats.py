import os
import json
import spacy
import statistics
import random
import operator
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.symbols import IS_PUNCT
import collections
# phrase_count = collections.Counter(caption_nes_examples[type]).most_common(100)

# statistics for reddit dataset
  
def NE_split(data):
    # filter and return samples where caption contains NEs
    data_with_ne = []
    data_without_ne = []
    print('Splitting data based on NE......')
    
    temp_count = 0
    for doc in data:
        if len(doc['text'].split()) == 0 or len(doc['caption'].split()) == 0:
            global removed_posts
            removed_posts += 1
            continue
        
        spacy_text = nlp(doc['text']) # NLP processed paragraph
        spacy_caption = nlp(doc['caption']) # NLP processed caption
        
        text_nes = [entity.text for entity in spacy_text.ents] # NEs in text
        caption_nes = [entity.text for entity in spacy_caption.ents] # NEs in caption
             
        if len(caption_nes) != 0: #caption contains NEs
            data_with_ne.append(doc)
        else:
            data_without_ne.append(doc)
        
        temp_count += 1
        print('Posts processed : {}/{}'.format(temp_count, 
                          len(data)), end='\r')
    
    print('Posts processed : {}/{}'.format(temp_count, len(data)))
    print('---------------------------------')
    return data_with_ne, data_without_ne
    
def typelift_split(data):
    # filter and return samples with NEs lifted to their types
    data_typelift = []
    
    print('Typelifting NEs in captions......')
    
    temp_count = 0
    for doc in data:
       
        if len(doc['text'].split()) == 0 or len(doc['caption'].split()) == 0:
            continue
        
        spacy_text = nlp(doc['text']) # NLP processed paragraph
        spacy_caption = nlp(doc['caption']) # NLP processed caption
        
        # ignore sample if caption contains named entity LAW - there are only a few such cases, not a major loss
        num_law = 0
        if 'LAW' in [entity.label_ for entity in spacy_caption.ents]:
            num_law += 1
            continue
        
        # lift NEs to their types based on heuristics
        direct_lifts = ['GPE', 'PERSON', 'ORG', 'NORP', 'FAC', 'WORK_OF_ART', 'PRODUCT']
        if_digit_lifts = ['CARDINAL', 'ORDINAL', 'QUANTITY']
        
        caption = doc['caption']
        for entity in spacy_caption.ents:
            caption = caption[:entity.start_char] + caption[entity.start_char:].replace(entity.text, '(' + entity.text + '/' + entity.label_ + ')', 1)
        
#         print('Original caption ({}): {}'.format(len(caption), caption))
        
        offset = 0
        offset_text = 0
        offset_label = 0
        for entity in spacy_caption.ents:
            bool_contains_digits = any(char.isdigit() for char in str(entity.text).strip())
            
#             print('\tEntity text: ', entity.text)
#             print('\tEntity label: ', entity.label_)
#             print('\tEntity start char: ', entity.start_char)
#             print('\tOffset: ', offset)
            
            if entity.label_ in direct_lifts:
                doc['caption'] = doc['caption'][:(entity.start_char - offset)] + doc['caption'][(entity.start_char - offset):].replace(entity.text, entity.label_, 1)
                offset_text = len(entity.text)
                offset_label = len(entity.label_)
            elif entity.label_ in if_digit_lifts and bool_contains_digits: # if string contains digit, typelift, otherwise not
                doc['caption'] = doc['caption'][:(entity.start_char - offset)] + doc['caption'][(entity.start_char - offset):].replace(entity.text, entity.label_, 1)
                offset_text = len(entity.text)
                offset_label = len(entity.label_)
            elif entity.label_ == 'DATE' and str(entity.text).isdigit(): # if all characters are digits, typelift, otherwise not
                doc['caption'] = doc['caption'][:(entity.start_char - offset)] + doc['caption'][(entity.start_char - offset):].replace(entity.text, entity.label_, 1)
                offset_text = len(entity.text)
                offset_label = len(entity.label_)
            elif entity.label_ == 'LOC' and "earth" not in str(entity.text).lower():
                doc['caption'] = doc['caption'][:(entity.start_char - offset)] + doc['caption'][(entity.start_char - offset):].replace(entity.text, entity.label_, 1)
                offset_text = len(entity.text)
                offset_label = len(entity.label_)
            elif entity.label_ == 'EVENT' and ("new year" not in str(entity.text).lower() or "christmas" not in str(entity.text).lower()):
                doc['caption'] = doc['caption'][:(entity.start_char - offset)] + doc['caption'][(entity.start_char - offset):].replace(entity.text, entity.label_, 1)
                offset_text = len(entity.text)
                offset_label = len(entity.label_)
            else:
                offset_text = 0
                offset_label = 0
            
            offset += (offset_text - offset_label)
#             print('Intermediate caption ({}): {}'.format(len(doc['caption']), doc['caption']))
            
        
#         print('Typelifted caption: ', doc['caption'])
#         print('---------------------------------')
        temp_count += 1
        print('Posts processed : {}/{}'.format(temp_count, len(data)), end='\r')
        data_typelift.append(doc)
    
    print('Posts processed : {}/{}'.format(temp_count, len(data)))
    print("Number of samples ignored because caption contains named entity LAW: ", num_law)
    print('---------------------------------')
    return data_typelift

def word_overlap_split(data, num_overlap):
    data_text_caption_overlap = []

    avg_overlap = 0
    
    temp_count = 0
    for doc in data:
       
        if len(doc['text'].split()) == 0 or len(doc['caption'].split()) == 0:
            continue
        
        spacy_text = nlp(doc['text']) # NLP processed paragraph
        spacy_caption = nlp(doc['caption']) # NLP processed caption
                  
        # text and caption tokens without stop words
        text_tokens = [token.text for token in spacy_text if not token.is_stop and not token.is_punct]
        caption_tokens = [token.text for token in spacy_caption if not token.is_stop and not token.is_punct]
        
        # calculate overlap (common tokens) between text and caption
        overlap = set(text_tokens).intersection(set(caption_tokens))
        
        if len(overlap) >= num_overlap:
            data_text_caption_overlap.append(doc)
        
        temp_count += 1
        print('Posts processed : {}/{}'.format(temp_count, len(data)), end='\r')
      
    print('Posts processed : {}/{}'.format(temp_count, len(data)), end='\r')
    print('---------------------------------')
    return data_text_caption_overlap
    
def basic_stats(data):
    with open('/GW/multimodal_embeddings/static00/reddit/praw/data/whole_processed_merged.json') as f: #whole_processed
        data = json.load(f)
        #     avg_text_len = statistics.mean(text_len)
#     mode_text_len = statistics.mode(text_len)
#     avg_caption_len = statistics.mean(caption_len)
#     mode_caption_len = statistics.mode(caption_len)
#     
#     NE_types_text_count = {}
#     for type in set(NE_types_text):
#         NE_types_text_count[type] = NE_types_text.count(type)
#     sorted_NE_types_text_count = dict(sorted(NE_types_text_count.items(), key=operator.itemgetter(1),reverse=True))
#        
#     NE_types_caption_count = {}
#     for type in set(NE_types_caption):
#         NE_types_caption_count[type] = NE_types_caption.count(type)
#     sorted_NE_types_caption_count = dict(sorted(NE_types_caption_count.items(), key=operator.itemgetter(1),reverse=True))
        
    
def main():
    # call various functions
    global nlp 
    nlp = spacy.load("en_core_web_sm")
    global removed_posts
    removed_posts = 0
    
    
    with open('/GW/multimodal_embeddings/static00/reddit/praw/data/whole_processed_merged.json') as f: #whole_processed_merged
        data = json.load(f)     
        
        data_with_ne,  data_without_ne = NE_split(data)
        json.dump(data_with_ne, 
                  open('/GW/multimodal_embeddings/static00/reddit/praw/data/whole_processed_with_ne.json', 'w+', encoding='utf8'), 
                  ensure_ascii=False)
        json.dump(data_without_ne, 
                  open('/GW/multimodal_embeddings/static00/reddit/praw/data/whole_processed_without_ne.json', 'w+', encoding='utf8'), 
                  ensure_ascii=False)
        
        data_typelift = typelift_split(data_with_ne)
        json.dump(data_typelift, 
                  open('/GW/multimodal_embeddings/static00/reddit/praw/data/whole_processed_typelift.json', 'w+', encoding='utf8'), 
                  ensure_ascii=False)
        
        print('Filtering samples with one word overlap for data with NE......')
        data_with_ne_one_overlap = word_overlap_split(data_with_ne, 1)
        json.dump(data_with_ne_one_overlap, 
                  open('/GW/multimodal_embeddings/static00/reddit/praw/data/whole_processed_with_ne_one_overlap.json', 'w+', encoding='utf8'), 
                  ensure_ascii=False)
        print('Filtering samples with two words overlap for data with NE......')
        data_with_ne_two_overlap = word_overlap_split(data_with_ne, 2)
        json.dump(data_with_ne_two_overlap, 
                  open('/GW/multimodal_embeddings/static00/reddit/praw/data/whole_processed_with_ne_two_overlap.json', 'w+', encoding='utf8'), 
                  ensure_ascii=False)
        
        print('Filtering samples with one word overlap for data without NE......')
        data_without_ne_one_overlap = word_overlap_split(data_without_ne, 1)
        json.dump(data_without_ne_one_overlap, 
                  open('/GW/multimodal_embeddings/static00/reddit/praw/data/whole_processed_without_ne_one_overlap.json', 'w+', encoding='utf8'), 
                  ensure_ascii=False)
        print('Filtering samples with two words overlap for data without NE......')
        data_without_ne_two_overlap = word_overlap_split(data_without_ne, 2)
        json.dump(data_without_ne_two_overlap, 
                  open('/GW/multimodal_embeddings/static00/reddit/praw/data/whole_processed_without_ne_two_overlap.json', 'w+', encoding='utf8'), 
                  ensure_ascii=False)
        
        print('Filtering samples with one word overlap for data with NE type-lifted......')
        data_typelift_one_overlap = word_overlap_split(data_typelift, 1)
        json.dump(data_typelift_one_overlap, 
                  open('/GW/multimodal_embeddings/static00/reddit/praw/data/whole_processed_typelift_one_overlap.json', 'w+', encoding='utf8'), 
                  ensure_ascii=False)
        print('Filtering samples with two words overlap for data with NE type-lifted......')
        data_typelift_two_overlap = word_overlap_split(data_typelift, 2)
        json.dump(data_typelift_two_overlap, 
                  open('/GW/multimodal_embeddings/static00/reddit/praw/data/whole_processed_typelift_two_overlap.json', 'w+', encoding='utf8'), 
                  ensure_ascii=False)
        
        #------------------------------------------------------------------------

        print("Samples with NE in caption: {} ({})".format(len(data_with_ne), (len(data_with_ne)/(len(data) - removed_posts))*100))
        print("\t\t One word overlap: {} ({})".format(len(data_with_ne_one_overlap), (len(data_with_ne_one_overlap)/(len(data) - removed_posts))*100))
        print("\t\t Two words overlap: {} ({})".format(len(data_with_ne_two_overlap), (len(data_with_ne_two_overlap)/(len(data) - removed_posts))*100))

        print("\nSamples with no NE in caption: {} ({})".format(len(data_without_ne), (len(data_without_ne)/(len(data) - removed_posts))*100))
        print("\t\t One word overlap: {} ({})".format(len(data_without_ne_one_overlap), (len(data_without_ne_one_overlap)/(len(data) - removed_posts))*100))
        print("\t\t Two words overlap: {} ({})".format(len(data_without_ne_two_overlap), (len(data_without_ne_two_overlap)/(len(data) - removed_posts))*100))

        print("\nSamples with NE in caption typelifted: {} ({})".format(len(data_typelift), (len(data_typelift)/(len(data) - removed_posts))*100))
        print("\t\t One word overlap: {} ({})".format(len(data_typelift_one_overlap), (len(data_typelift_one_overlap)/(len(data) - removed_posts))*100))
        print("\t\t Two words overlap: {} ({})".format(len(data_typelift_two_overlap), (len(data_typelift_two_overlap)/(len(data) - removed_posts))*100))

    return True

if __name__ == '__main__':
    main()