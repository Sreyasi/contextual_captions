# torch specific
import torch
import torch.cuda as cuda

# model specific
import model_utils
from models.ShowTellModel import ShowTellModel
import torch.optim as optim
import torch.nn as nn

# data specific
from params import parse_args
from data_utils import Dataset, collate_fn
from data_utils import Vocabulary
from torch.utils.data import DataLoader
import pprint

# others
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
from tqdm import tqdm
import json
import pickle
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import copy
import pdb


def init_config():
    # read the config file
    global config
    config = json.load(open(args.config_file, 'r'))
    
    global data_dir#, image_data_dir
    data_dir = config['data_dir']
#     image_data_dir = config['image_data_dir']
    
    global train_data_file, test_data_file, val_data_file, whole_data_file
    train_data_file = config['train_data_file']
    test_data_file = config['test_data_file']
    val_data_file = config['val_data_file']
    whole_data_file = config['whole_data_file']
    
    # used for all .random invokes
    global SEED, USE_CUDA, LongTensor, FloatTensor    
    SEED = 88
    if cuda.is_available():
        USE_CUDA = True
        LongTensor = torch.cuda.LongTensor
        FloatTensor = torch.cuda.FloatTensor
    else:
        USE_CUDA = False
        LongTensor = torch.LongTensor
        FloatTensor = torch.FloatTensor
    return None

def init_vocab(filename):
            
    with open(filename, 'r') as f:
        data = json.load(f, encoding='utf8')
    fout = './data/vocab.pickle'
    # Intialize the vocabulary or load generated vocab
    if args.preprocess:
        v = Vocabulary()
        vocab = v.build_vocab(filename)
        pickle.dump(vocab, open(data_dir + fout, 'wb'))
    else:
        vocab = pickle.load(open(data_dir + fout, 'rb'))

    print("Vocabulary size = {}".format(len(vocab['vocab'])))
    config['vocab_size'] = len(vocab['vocab'])
    
    if args.eval == '' and args.resume == '':
        # get save directory
        curtime = datetime.now()
        timestamp = curtime.strftime('%Y_%m_%d_%H_%M_%S')
        savedir = '{}run{}_{}_lonelyPlanet_bert'.format(args.savedir, str(len(data)), timestamp)
        
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        # save arguments 
        if args.train:
            argspath = os.path.join(savedir, 'gen_args-1.json')
        if not os.path.exists(argspath):
            with open(argspath, 'w') as f:
                json.dump(args.__dict__, f, indent=2)
    else:
        savedir = args.savedir
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        
    return vocab, savedir


def get_datagen(vocab, tokenizer):
    
    train_set = Dataset(os.path.join(data_dir, train_data_file), vocab, tokenizer)
    train_generator = DataLoader(
            train_set, batch_size = config['train_batch_size'],
            shuffle=True, collate_fn= collate_fn, 
            num_workers=config['num_workers'])
    
    val_set = Dataset(os.path.join(data_dir, val_data_file), vocab, tokenizer)
    val_generator = DataLoader(
            val_set, batch_size = config['dev_batch_size'],
            shuffle=False, collate_fn= collate_fn, 
            num_workers=config['num_workers'])
    
    test_set = Dataset(os.path.join(data_dir, test_data_file), vocab, tokenizer)
    test_generator = DataLoader(
            test_set, batch_size = config['dev_batch_size'],
            shuffle=False, collate_fn= collate_fn, 
            num_workers=config['num_workers'])
    
    print('No. of train samples: {}'.format(len(train_set)))
    print('No. of val samples: {}'.format(len(val_set)))
    print('No. of test samples: {}'.format(len(test_set)))
    return train_generator, val_generator, test_generator


def init_models(vocab):
    
    if args.gen_type == 'showtell':
        g_net = ShowTellModel(config, vocab)
        global g_optim, nll_criterion
        g_optim = optim.Adam(g_net.parameters(), lr=config['lr'])
        nll_criterion = nn.NLLLoss()
    else:
        raise ValueError('only showtell supported for now!')
        
    if USE_CUDA: 
        g_net = g_net.cuda()
    
    return g_net,  g_optim


def train_generator(g_net, traindata, pretrained_bert=None, tokenizer=None):

    image_files, img_vec, gt_caps, paragraph = traindata
    if args.gen_type == 'ourshowtell':
        batch_loss = g_net((img_vec, gt_caps, paragraph), args)
    elif args.gen_type == 'showtell':
        logprobs = g_net.forward(img_vec, gt_caps, paragraph, pretrained_bert)
        _, seq_len, _ = logprobs.size()
        target = gt_caps[:, 1:].reshape(-1) # Shift the target by 1 place
        batch_loss = nll_criterion(logprobs.view(-1, config['vocab_size']), target)
        # backpropgate the loss
        g_optim.zero_grad()
        batch_loss.backward()
        g_optim.step()
    else:
        raise ValueError('only showtell and ourshowtell returned')
        
    return batch_loss


# ***** Main function to train the model ********
def train_model(train_data_generator, g_net, g_epoch, giter,  gwriter,  vocab, pretrained_bert=None, tokenizer=None):
    
    num_batches = 0
    epoch_loss_g = 0.0
    for (imagefiles, batch_images, batch_abstract,
         batch_paragraph) in tqdm(train_data_generator):

        # use cuda
        if USE_CUDA:
            batch_images = batch_images.cuda()
            batch_abstract = batch_abstract.cuda()
            batch_paragraph = batch_paragraph.cuda()

        train_data = (imagefiles, batch_images, batch_abstract, batch_paragraph) #batch_unpaired

        
        # train model
        if args.train:
            batch_loss_g = train_generator(g_net, train_data, pretrained_bert, tokenizer)
            giter += 1 
            
        epoch_loss_g += batch_loss_g
        num_batches += 1

        # write losses to tensorboard for plots
        if args.train:
            gwriter.add_scalar('train_g_loss', batch_loss_g,
                              giter)

    epoch_loss_g = epoch_loss_g / num_batches
        
    return epoch_loss_g, giter, gwriter
    

# ***** Inference for the GAN model ********
def val_model(g_net, data_generator, save_dir, epoch, vocab, 
              batchsize, writer, valiter=None, pretrained_bert=None, tokenizer=None):
    
    tot_val_loss = 0.0    
    num_batches = 0
    if args.eval != '':
        output_txt = open(os.path.join(
            #TODO save with timestamp
            save_dir, "generated_captions_e" + str(epoch) + "_bert.txt"), "w+", 
            encoding='utf-8')
        output_json = os.path.join(save_dir, 
                                   "test_results_e" + str(epoch) + "_bert.json")
    else:
        output_txt = open(os.path.join(
            save_dir, "generated_captions_val_e" + str(epoch) + ".txt"), "w+", 
            encoding='utf-8')
        output_json = os.path.join(save_dir, 
                                   "val_results_e" + str(epoch) + ".json")
    output_data = []
    
#     data_generator = list(filter(lambda x : x is not None, data_generator))
    
#     for (batch_images_file_names, batch_image_feats, batch_abstract, batch_paragraph, batch_noun_pos, batch_ner_pos) in data_generator:
#     for (batch_images_file_names, batch_image_feats, batch_abstract, batch_paragraph, batch_noun_pos) in data_generator:
    for (batch_images_file_names, batch_image_feats, batch_abstract, batch_paragraph) in data_generator:

        if USE_CUDA:
            batch_image_feats = batch_image_feats.cuda()
            batch_abstract = batch_abstract.cuda()
            batch_paragraph = batch_paragraph.cuda()
#             batch_noun_pos = batch_noun_pos.cuda()
#             batch_ner_pos = batch_ner_pos.cuda()
            
            if args.greedy:
                if args.gen_type == 'ourshowtell':
                    gen_seq, val_loss = g_net.greedy_inference(
                            batch_image_feats, batch_abstract,
                            batchsize, batch_paragraph)
                    tot_val_loss += val_loss
                elif args.gen_type == 'showtell':
#                     gen_seq, gen_seq_logprob = g_net.sample(batch_image_feats, batch_paragraph, batch_noun_pos, batch_ner_pos)
#                     gen_seq, gen_seq_logprob = g_net.sample(batch_image_feats, batch_paragraph, batch_noun_pos)
                    gen_seq, gen_seq_logprob = g_net.sample(batch_image_feats, batch_paragraph, pretrained_bert=pretrained_bert, tokenizer=tokenizer)
                    val_loss = -torch.mean(gen_seq_logprob)
                    tot_val_loss += val_loss
            elif args.beam:
                if args.gen_type == 'ourshowtell':
                    gen_seq = g_net.beam_search_inference(
                            batch_image_feats, batch_abstract,
                            batchsize, batch_paragraph)
                elif args.gen_type == 'showtell':
                    gen_seq, gen_seq_logprob = g_net.sample_beam(batch_image_feats, batch_paragraph, beam_size=5)
            else:
                raise ValueError('did not give inference type')
   
        print("IMAGE: ", batch_images_file_names[0])
        
        paragraph = batch_paragraph.cpu().numpy()[:, 1:].tolist()        
        if args.use_bert_tokenizer:
            paragraph = tokenizer.decode(paragraph[0], skip_special_tokens=True)
            print("TEXT: ", paragraph)
        else:
            paragraph = model_utils.vec_2_text(paragraph, vocab['i2w'])
            print("TEXT: ", paragraph[0])

        gen_seq = gen_seq.cpu().numpy().tolist()
        if args.use_bert_tokenizer:
            gen_seq = tokenizer.decode(gen_seq[0], skip_special_tokens=True)
            print("GENERATED: ", gen_seq)
        else:
            gen_seq = model_utils.vec_2_text(gen_seq, vocab['i2w'])
            print("GENERATED: ", gen_seq[0])

        gt_abstract = batch_abstract.cpu().numpy()[:, 1:].tolist()
        if args.use_bert_tokenizer:
            gt_abstract = tokenizer.decode(gt_abstract[0], skip_special_tokens=True)
            print("GROUND TRUTH: ", gt_abstract)
        else:
            gt_abstract = model_utils.vec_2_text(gt_abstract, vocab['i2w'])
            print("GROUND TRUTH: ", gt_abstract[0])

        output_txt.write(
                batch_images_file_names[0] + "\t" +  
                paragraph +
                "\t" + gt_abstract +
                "\t" + gen_seq + "\n")

        print("-----------------------------------------------")
            
        
        if valiter is not None:
            writer.add_scalar('val_g_loss', val_loss, valiter)
            valiter += 1
        num_batches += 1
            
        doc = {}
        doc['image'] = batch_images_file_names[0]
        doc['text'] = paragraph
        doc['generated'] = gen_seq
        doc['gt'] = gt_abstract
        
        output_data.append(doc)
        
    fout = open(output_json, 'w', encoding='utf-8')
    json.dump(output_data, fout)
        
    output_txt.close()
    fout.flush()
    fout.close()
    print("VAL LOSS: ", tot_val_loss/num_batches)
    if valiter is not None:
        return tot_val_loss/num_batches, valiter
    else:
        return tot_val_loss/num_batches

# ***** Main function ********
def main(flags):
    
    global args
    args = copy.deepcopy(flags)
    
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(args.__dict__)
    
    init_config()
    
    vocab, save_dir = init_vocab(os.path.join(data_dir, whole_data_file))

    #vocab = []
    #save_dir = args.savedir

    if args.use_bert_tokenizer:
        # Import pre-trained BERT tokenizer
        from transformers import BertTokenizer, BertModel
        if args.train:
            print("INFO: The model will use BERT Tokenizer.")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            special_tokens_dict = {'bos_token': '<sos>', 'eos_token': '<eos>'}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            print("INFO: {} new token added to the tokenizer".format(num_added_toks))
            config['vocab_size'] = len(tokenizer)
            print("INFO: Token vocabulary size ==> {}".format(len(tokenizer)))
            pretrained_bert = BertModel.from_pretrained('bert-base-uncased')
            pretrained_bert = pretrained_bert.cuda()
            pretrained_bert.resize_token_embeddings(len(tokenizer))

            # Save and Freeze the tokenizer and the pretrained BERT
            pretrained_bert.save_pretrained(args.savedir)
            tokenizer.save_pretrained(args.savedir)
        elif args.eval:
            tokenizer = BertTokenizer.from_pretrained(args.savedir)
            pretrained_bert = BertModel.from_pretrained(args.savedir)
            pretrained_bert = pretrained_bert.cuda()
            pretrained_bert.resize_token_embeddings(len(tokenizer))
            config['vocab_size'] = len(tokenizer)
    else:
        tokenizer = None

    
    # initalize training, val and test data generators
    train_generator, val_generator, test_generator = get_datagen(vocab, tokenizer)
    
    # get the models, optimizers, config initialized
    g_net,  g_optim = init_models(vocab)

    # Print the trainable parameters
    print("***** Trainable params ****")
    for name, p in g_net.named_parameters():
        print(name, p.requires_grad)
    
    np.random.seed(SEED)
    random.seed(SEED)
    # load existing model and optimizer states onto the initialized ones
    # get the previously ended epoch and iter as well
    if args.eval != '' or args.resume != '':
        (gen_stuff, save_dir) = model_utils._load_model(args, g_net, config)
        g_net, g_epoch, g_niter, g_val_epoch, g_val_iter = gen_stuff

    else:
        g_epoch = 1
        g_val_epoch = 1
        g_niter = 0
        g_val_iter = 0
    g_writer = SummaryWriter(save_dir)
        
    print('saving directory: {}'.format(save_dir))
    
    if args.eval != '':
        print('inference mode')

        if args.use_bert_tokenizer:
            _, _ = val_model(g_net, test_generator, save_dir, g_epoch, vocab,
                             config['dev_batch_size'], g_writer, pretrained_bert=pretrained_bert, tokenizer=tokenizer)
        else:
            _,_ = val_model(g_net, test_generator, save_dir, g_epoch, vocab,
                        config['dev_batch_size'], g_writer)

    else:
        if args.train:
            print("Training generator network")
        else:
            raise ValueError('none of the modes are ON. Check..')

        for epoch in range(1, int(args.epoch) + 1):
            print('==> Training:')
            print("Epoch = {}".format(epoch))
            
            # train model for 1 epoch
            if args.use_bert_tokenizer:
                (epoch_loss_g, g_niter, g_writer) = train_model(train_generator, g_net,
                                                   g_epoch,
                                                   g_niter,
                                                   g_writer,
                                                   vocab, pretrained_bert, tokenizer)
            else:
                (epoch_loss_g, g_niter, g_writer) = train_model(train_generator, g_net,
                                                                g_epoch,
                                                                g_niter,
                                                                g_writer,
                                                                vocab)
                # validate model every val_freq epochs
            # if epoch % args.val_freq == 0 and (args.train):
            #
            #     g_val_epoch += 1
            #     print('-' * 60)
            #     print('==> Val:')
            #     print("Epoch = {}".format(epoch))
            #
            #     g_valloss, g_val_iter = val_model(
            #             g_net, val_generator, save_dir, g_epoch, vocab,
            #             config['dev_batch_size'], g_writer, g_val_iter)
            #
            #     g_writer.add_scalar('val_g_epoch_loss', g_valloss,
            #                         g_val_epoch)

            # save model every save_freq epochs and at last epoch
            if epoch % args.save_freq == 0 or epoch == int(args.epoch):

                model_utils._save_model(args, g_net,
                                        g_optim,
                                        g_epoch,
                                        g_val_epoch,
                                        g_niter,
                                        g_val_iter,
                                        save_dir)
            
            # print loss
            print("GenEpoch = {}, Gen Loss = {}".format(g_epoch, epoch_loss_g))
            
            # log loss values and increase epoch number by 1
            if args.train:
                g_writer.add_scalar('train_g_epoch_loss', epoch_loss_g,
                                  g_epoch)
                g_epoch += 1
                
    g_writer.close()
            
    return save_dir, g_epoch

if __name__ == '__main__':

    args = parse_args()
    main(args)
