import json
import random
from sklearn.utils import shuffle
from torch import nn
from torchvision import models
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import torch
import pdb
import re
from unicodedata import normalize

import argparse
config = json.load(open('config.json', 'r'))   

class ImageEncoder(nn.Module):
    def __init__(self, modeltype):
        """Load the pretrained model and replace top fc layer."""
        super(ImageEncoder, self).__init__()
        if modeltype == 'resnet152':
            self.ImageEnc = models.resnet152(pretrained=True)
        elif modeltype == 'resnet101':
            self.ImageEnc = models.resnet101(pretrained=True)
        elif modeltype == 'resnet50':
            self.ImageEnc = models.resnet50(pretrained=True)
        elif modeltype == 'resnet18':
            self.ImageEnc = models.resnet18(pretrained=True)
        else:
            raise ValueError('{} not supported'.format(modeltype))
        self.layer = self.ImageEnc._modules.get('avgpool')
        self.ImageEnc.eval()
    
    
    def forward(self, images):
        """Extract the image feature vectors."""
        my_embedding = torch.zeros(1, 2048)
        def copy_data(m, i, o):
            my_embedding.copy_(torch.flatten(o.data, 1))
        h = self.layer.register_forward_hook(copy_data)
        self.ImageEnc(images)
        h.remove()
            
        return my_embedding
    
def extract_feats(img_id, cnnmodel):
        
    isvalid = False
    image_file = os.path.join(config['image_data_dir'], img_id)
    if os.path.exists(image_file) and os.path.getsize(image_file)/1024 < 30:
        return None, isvalid
    try:
        image = Image.open(image_file).convert('RGB')
        image_loader = transforms.Compose(
                    [transforms.Resize(config["input_image_dim"]),
                    transforms.CenterCrop(224), transforms.ToTensor(), 
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])])
            
        if image_loader(image).float().shape[0] == 1:
            image_vec = torch.from_numpy(
                    np.stack((image_loader(image).float())*3))
            print("Converted 1 channel image to 3 channel image.", 
                  image_vec.shape)
        else:
            image_vec = image_loader(image).float()
            
        image_vec = image_vec.cuda()
        featvec = cnnmodel(image_vec.unsqueeze(0)).cpu().numpy()
        isvalid = True
    except:
        featvec = None
        pass
    return featvec, isvalid
 
def get_data(removedimages, numsamples, seenimages):
    print('currently {} images already seen.'.format(len(seenimages)))
    output_data = []
    curr_seen_images = set()
    with open('/GW/multimodal_embeddings/static00/reddit/praw/downloads/whole_data.json', 'r') as f:
        data = json.load(f)
        
        # shuffle data
        data = shuffle(data)
        
        count = 0
        totcount = len(data)
        removecount = 0
#         text_length_count_less_50 = 0
#         text_length_count_greater_500 = 0
        avg_text_length = 0
        avg_cap_length = 0
        mod_text_count = 0
        
        for doc in data:
            image_hash = doc['image_hash']
            text = doc['text']
            caption = doc['caption']
            
            # do not consider samples with broken images
            if image_hash in removedimages:
                removecount += 1
                continue
            
            # do not consider samples in other splits (train/test/val)
            if image_hash in seenimages:
                removecount += 1
                continue            
            
            # do not consider samples with template moderator message
            if "This post has been automatically removed" in text or "Unfortunately, it has been removed for violating" in text:
#                 print("TEMPLATE MODERATOR MESSAGE!! REMOVING POST!!")
                mod_text_count += 1
                removecount += 1
                removedimages.append(image_hash)
                continue
            
#             if len(text.split()) < 50 or len(text.split()) > 500:
            if len(text.split()) == 0 or text == '':
                removecount += 1 # print("Too small text. Image not considered.")
#                 if len(text.split()) < 50:
#                     text_length_count_less_50 += 1
#                 else:
#                     text_length_count_greater_500 += 1
                removedimages.append(image_hash)
                continue
            
#             print("Image: ",  image_hash)
#             print("Original text: ", text)
#             print("Original text length: ", len(text.split()))
            
            # remove urls
            text = re.sub(r'https?://\S+', '',  text)
            caption = re.sub(r'https?://\S+', '',  caption)
            
            # remove new lines
            text = re.sub(r'\n', '', text)
            caption = re.sub(r'\n', '', caption)
            
            # handle \u...
            text = re.sub(r'\\u2019', "'", text)
            caption = re.sub(r'\\u2019', "'", caption)
            #text = doc['text'] # u"".join(doc['text'])
            #caption = doc['caption'] # u"".join(doc['caption'])
            mod_text = normalize('NFKD', text).encode('ascii','ignore')
            mod_caption = normalize('NFKD', caption).encode('ascii','ignore')
            doc['text'] = mod_text.decode("utf-8")
            doc['caption'] = mod_caption.decode("utf-8")
            
#             print("Modified text: ", doc['text'])
            
            # Space before punctuations for captions
            doc['caption'] = doc['caption'].replace(
                    ',', ' ,').replace('.', ' .').replace(
                            ':', ' :').replace(';', ' ;').replace(
                                    '?', ' ?').replace('!', ' !')
            
#             print("Modified caption: ", doc['caption'])
            
            curr_seen_images.add(image_hash)
            
            if doc['caption'] != '' and doc['text'] != '':
                output_data.append(doc)
                count += 1
                avg_text_length += len(doc['text'].split())
                avg_cap_length += len(doc['caption'].split())
#                 print("Image considered.")
#                 print("\n_______________________________________")
            if numsamples is not None:
                if count == numsamples:
                    break
            print('num samples processed: {}/{}'.format(count, totcount), 
                  end='\r')
        print('num samples processed: {}/{}'.format(count, totcount))
        print('no of samples removed: {}'.format(removecount))
        print('no. of samples with template bot text: ', mod_text_count)
#         print('\nno of samples with text less than 50 words: ', text_length_count_less_70)
#         print('\nno of samples with text greater than 500 words: ', text_length_count_greater_500)
        print('average text length: ', (avg_text_length/count))
        print('average caption length: ', (avg_cap_length/count))
    return output_data, curr_seen_images, removedimages  
            
def main_func(flags):
     
    if flags.num_samples == 100000:
        num_samples = [100000, 5000, 5000]
    elif flags.num_samples == 10:
        num_samples = [10, 5, 5]
    elif flags.num_samples == 0:
        num_samples = [None, None, None]
        
#     process = ['train', 'val', 'test']
    process = ['whole']
      
    if flags.extract_feat:
        print('extracting image features...')
        # extracting features for all downloaded images with a significant amount of surrounding text
        process = ['whole']
        
        # init image features stuff
        modeltype = flags.modeltype
        imgfeatsavedir = "/GW/multimodal_embeddings/static00/reddit/praw/image_feats/" + modeltype + "/"
        cnnmodel = ImageEncoder(modeltype)
        cnnmodel = cnnmodel.cuda()
        print('image encoder {} loaded..'.format(modeltype))
          
        if not os.path.exists(imgfeatsavedir):
            os.makedirs(imgfeatsavedir)
          
        # Extract features and store images that caused errors
        allimages = set()
        removedimages = set()
        
        # for lonely planet
#         with open('/GW/multimodal_embeddings/static00/reddit/praw/data/lonelyPlanet/test_100.json', 'r') as f:
#         data = json.load(f)
#         totcnt = len(data)
#         tempcnt = 0
#         for doc in data:
#             image_hash = doc['image_hash']
#             print(image_hash)
#             if image_hash not in allimages and image_hash not in removedimages:
#                 output_feat, isvalid = extract_feats(image_hash, cnnmodel)
#                                
#                 if isvalid:
#                     allimages.add(image_hash)
#                     tempcnt += 1
#                     print("Saving image features.")
#                     np.save(imgfeatsavedir + image_hash + '.npy', output_feat)                                 
#                 else:
#                     removedimages.add(image_hash)
#                     print('Valid images: {}, invalid images: {}'.format(
#                     len(allimages), len(removedimages)), end='\r')
#             else:
#                 pass
#         print('no of images for Lonely Planet : {}/{}'.format(tempcnt, totcnt))
        
        # all jsons files from reddit scrape
        print('loading existing jsons...')
        json_path = '/GW/multimodal_embeddings/static00/reddit/praw/downloads/jsons'
        jsons = [f for f in os.listdir(json_path) if os.path.isfile(os.path.join(json_path, f))]
        # saving this list so as not to process them later again
        with open('/GW/multimodal_embeddings/static00/reddit/praw/downloads/processed_jsons_2.txt', 'w') as jsons_processed:
            jsons_processed.write('\n'.join(jsons))
            
        print('reading {} json files from reddit scrape.'.format(len(jsons)))
        
        all_json = []      
      
        for idx in range(len(process)):
              
            proc = process[idx]
            print('extracting features for {} set..'.format(proc))
            #with open('./data/personality_captions/' + proc + '.json', 'r') as f:
            # '/GW/D5data-12/sreyasi/contextual_captions/data/lonelyPlanet_imageDetails_sameFormat.json'
            totcnt = len(jsons)
            tempcnt = 0
            for json_file in jsons:
                #print(os.path.join(json_path, json_file))
                with open(os.path.join(json_path, json_file), 'r') as f:
                    data = json.load(f)
                    #for doc in data:
                    # doc contains comments, url, title, id
                    #print(doc)
                    url = data['url']
                    image_hash = os.path.basename(url)
                    caption = data['title']
                    text = data['comments']
                    id = data['id']
                        
                    # check if text contain 80-200 words
#                     if len(text.split()) < 80 and len(text.split()) > 200:
#                         break

                    if image_hash not in allimages and image_hash not in removedimages:
                              
                        if os.path.exists(os.path.join(imgfeatsavedir, image_hash + '.npy')):
                            isvalid = True
                        else:
                            output_feat, isvalid = extract_feats(image_hash, 
                                                                 cnnmodel)
                              
                        if isvalid:
                            allimages.add(image_hash)
                            tempcnt += 1
                            # save features only for new images
                            if not os.path.exists(os.path.join(imgfeatsavedir, image_hash + '.npy')):
                                np.save(imgfeatsavedir + image_hash + '.npy', 
                                        output_feat)
                            new_doc = {}
                            new_doc['id'] = id
                            new_doc['image_hash'] = image_hash
                            new_doc['caption'] = caption
                            new_doc['text'] = text
                                
                            all_json.append(new_doc)
                        else:
                            removedimages.add(image_hash)
              
                        #print('Valid images: {}, invalid images: {}'.format(
                        #        len(allimages), len(removedimages)), end='\r')
                    else:
                        pass
                    
                    print('no of images for {} : {}/{}'.format(proc, tempcnt, 
                          totcnt), end='\r')
        print('Total no of Valid images: {}, invalid images: {}'.format(
                len(allimages), len(removedimages)))
        
        # store data for all the jsons processed and corresponding image features extracted
        all_json_out = open('/GW/multimodal_embeddings/static00/reddit/praw/downloads/whole_data.json', 'w+', encoding='utf8')
        json.dump(all_json, all_json_out)

        json.dump(list(allimages), 
                   open('/GW/multimodal_embeddings/static00/reddit/praw/downloads/processed_images.json', 'w+', encoding='utf8'))
        json.dump(list(removedimages), 
                  open('/GW/multimodal_embeddings/static00/reddit/praw/downloads/removed_images.json', 'w+', encoding='utf8'))
    else:
          
        removedimages = json.load(
                open('/GW/multimodal_embeddings/static00/reddit/praw/downloads/removed_images.json', 'r'))

        # prepare data
        prev_output_data = []
        seenimages = set()
        
        for idx in range(len(process)):
              
            proc = process[idx]
            print('------------------------\nprocessing {} set.'.format(proc))
            numsamp = num_samples[idx]
            if numsamp is not None:
                op_file = proc + '_' + str(numsamp)
#                 ip_file = proc 
            else:
                op_file = proc + '_processed'
#                 ip_file = proc
            
            fout = open('/GW/multimodal_embeddings/static00/reddit/praw/data/' + op_file + '.json', 'w+', encoding='utf8')
      
            # store data selected in each idx (train/val/test) and remove them from orig data
            output_data, prev_seen_images, removedimages = get_data(removedimages, numsamp, seenimages)
              
            json.dump(output_data, fout, ensure_ascii=False)
            
            #with open('/GW/D5data-12/sreyasi/contextual_captions/data/' + op_file + '.json', 'r') as f_prev:
             #     prev_data = json.load(f_prev)
              #    for prev_doc in prev_data:
               #         prev_image_hash = prev_doc['image_hash']
                #        seenimages.add(prev_image_hash)
            
            print('data for {} extracted: {} samples'.format(proc, 
                  len(output_data)))
                  
            seenimages.update(prev_seen_images)
            
            #print('number of seen images: ', len(seenimages))
    
    return True
     
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='aumgent data parameters')
    parser.add_argument('--extract_feat', action='store_true', 
                        help='do we need to extract features for all images?')
    parser.add_argument('--modeltype', default='resnet152', type=str,
                        help='which model for image features?')
    parser.add_argument('--process_data', action='store_true', 
                        help='are we using bert to get word vectors?')
    parser.add_argument('--num_samples', default=0, type=int)
    
    flags = parser.parse_args()
    main_func(flags)
         
    