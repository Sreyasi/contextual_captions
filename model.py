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
    fout = '/vocab_' + str(len(data)) + '.pickle'
    # Intialize the vocabulary or load generated vocab
    if args.preprocess:
        v = Vocabulary()
        vocab = v.build_vocab(filename)
        pickle.dump(vocab, open(data_dir + fout, 'wb'))
    else:
        vocab = pickle.load(open(data_dir + fout, 'rb'))

    print("Vocabulary size = {}".format(len(vocab['vocab'])))
    config['vocab_size'] = len(vocab['vocab'])
    # print("# of Personalities = {}".format(len(vocab['p2i'])))
    
    if args.eval == '' and args.resume == '':
        # get save directory
        curtime = datetime.now()
        timestamp = curtime.strftime('%Y_%m_%d_%H_%M_%S')
        savedir = '{}run{}_{}_with_ne_30000_one_overlap'.format(args.savedir, str(len(data)), timestamp)
        
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        # save arguments 
        if args.pretrainG or args.train:
            argspath = os.path.join(savedir, 'gen_args-1.json')
        if not os.path.exists(argspath):
            with open(argspath, 'w') as f:
                json.dump(args.__dict__, f, indent=2)
        if args.pretrainD or args.train:
            argspath = os.path.join(savedir, 'dis_args-1.json')
        if not os.path.exists(argspath):
            with open(argspath, 'w') as f:
                json.dump(args.__dict__, f, indent=2)
    else:
        savedir = args.savedir
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        
    return vocab, savedir


def get_datagen(vocab):
    
    train_set = Dataset(os.path.join(data_dir, train_data_file), vocab)
    train_generator = DataLoader(
            train_set, batch_size = config['train_batch_size'],
            shuffle=True, collate_fn= collate_fn, 
            num_workers=config['num_workers'])
    
    val_set = Dataset(os.path.join(data_dir, val_data_file), vocab)
    val_generator = DataLoader(
            val_set, batch_size = config['dev_batch_size'],
            shuffle=False, collate_fn= collate_fn, 
            num_workers=config['num_workers'])
    
    test_set = Dataset(os.path.join(data_dir, test_data_file), vocab)
    test_generator = DataLoader(
            test_set, batch_size = config['dev_batch_size'],
            shuffle=False, collate_fn= collate_fn, 
            num_workers=config['num_workers'])
    
    print('No. of train samples: {}'.format(len(train_set)))
    print('No. of val samples: {}'.format(len(val_set)))
    print('No. of test samples: {}'.format(len(test_set)))
    return train_generator, val_generator, test_generator


def init_models(vocab):
       
    # initialize generator and discriminator models

    # if args.dis_type == 'cnn':
    #     d_net = CNNDiscriminator(vocab)
    # elif args.dis_type == 'rnn':
    #     d_net = RNNDiscriminator(vocab)
    # elif args.dis_type == 'none':
    #     d_net = None
    #     print('Discriminator type is NONE.')
    # else:
    #     raise ValueError('only cnn or rnn discriminators supported')
    
    if args.gen_type == 'showtell':
        g_net = ShowTellModel(config, vocab)
        global g_optim, nll_criterion #, adv_criterion
        g_optim = optim.Adam(g_net.parameters(), lr=config['lr'])
        nll_criterion = nn.NLLLoss() # ignore_index=0
        # adv_criterion = AdversarialLoss()
    # elif args.gen_type == 'ourshowtell':
    #     g_net = Generator(vocab)
    #     g_net.set_optim(lr=config['lr'], optimizer='Adam')
    else:
        raise ValueError('only showtell supported for now!')
        
    if USE_CUDA: 
        g_net = g_net.cuda()
    # if USE_CUDA and args.dis_type != 'none':
    #     d_net = d_net.cuda()

    # if args.dis_type != 'none':
    #     d_net.set_optim(lr=config['lr'], optimizer='Adam')

    # Initilize the language Evaluator
    # ref_file = os.path.join(config['data_dir'], config['train_data_file'])
    # lang_eval = LangEval(ref_file)
    
    # Initilize rollout
    # rollout = Rollout(g_net, 0.8)
    return g_net,  g_optim # d_net, lang_eval, rollout,


def pretrain_generator(g_net, traindata):
    '''
    Pretraining part of the gen using MLE 
    Args:
        g_net           : Generator object
        image_vec       : (batchsize, img_feat_dim) image vec
        gt_captions     : (batchsize, seqlen) of corresponding captions
        personalityinput: (batchsize, 1) of corresponding personality tokens
    Returns:
        loss            : generator loss for the input batch of samples
    '''
#     image_files, img_vec, gt_caps, paragraph, noun_pos, ner_pos = traindata
#     image_files, img_vec, gt_caps, paragraph, noun_pos = traindata
    image_files, img_vec, gt_caps, paragraph = traindata
    # print(paragraph.size())
    if args.gen_type == 'ourshowtell':
        batch_loss = g_net((img_vec, gt_caps, paragraph), args)
    elif args.gen_type == 'showtell':
#         logprobs = g_net.forward(img_vec, gt_caps, paragraph, noun_pos, ner_pos)
#         logprobs = g_net.forward(img_vec, gt_caps, paragraph, noun_pos)
        logprobs = g_net.forward(img_vec, gt_caps, paragraph)
        # print(logprobs.size())
        _, seq_len, _ = logprobs.size()
        target = gt_caps[:, 1:].reshape(-1) # Shift the target by 1 place
        batch_loss = nll_criterion(logprobs.view(-1, config['vocab_size']), target)
        # print(batch_loss)
        # backpropgate the loss
        g_optim.zero_grad()
        batch_loss.backward()
        g_optim.step()
    else:
        raise ValueError('only showtell and ourshowtell returned')
        
    return batch_loss


# ***** Main function to train the GAN model ********
def train_model(train_data_generator, g_net, g_epoch, giter,  gwriter,  vocab):
    '''
    Main function to train the GAN model
    Args:
        epoch            : epoch number
        batch_images     : (batchsize, channels, w, h) images
        batch_abstract   : (batchsize, seqlen) of corresponding captions
        abstract_lengths : (batchsize, 1) of actual length of sentences
        batch_personality: (batchsize, 1) of corresponding personality tokens
        g_net            : Generator model
        d_net            : Discriminator model
        rollout          : Rollout function
        args             : Bool object to determine the process
        writer           : tensorboard writer object 
    Returns:
        generator_loss    : loss of generator during pretraining/training
        discriminator_loss: loss of dis during pretraining/training
        
    '''
    num_batches = 0
    epoch_loss_g = 0.0
    epoch_loss_d = 0.0
#     for (imagefiles, batch_images, batch_abstract,
#          batch_paragraph, batch_noun_pos_vec, batch_ner_pos_vec) in tqdm(train_data_generator):
#     for (imagefiles, batch_images, batch_abstract,
#          batch_paragraph, batch_noun_pos_vec) in tqdm(train_data_generator):
    for (imagefiles, batch_images, batch_abstract,
         batch_paragraph) in tqdm(train_data_generator):

        # use cuda
        if USE_CUDA:
            batch_images = batch_images.cuda()
            batch_abstract = batch_abstract.cuda()
            # batch_unpaired = batch_unpaired.cuda()
            batch_paragraph = batch_paragraph.cuda()
#             batch_noun_pos_vec = batch_noun_pos_vec.cuda()
#             batch_ner_pos_vec = batch_ner_pos_vec.cuda()
        
#         train_data = (imagefiles, batch_images, batch_abstract, batch_paragraph, batch_noun_pos_vec, batch_ner_pos_vec) #batch_unpaired
#         train_data = (imagefiles, batch_images, batch_abstract, batch_paragraph, batch_noun_pos_vec) #batch_unpaired
        train_data = (imagefiles, batch_images, batch_abstract, batch_paragraph) #batch_unpaired

        
        # train model
        # Pre-Train Generator
        if args.pretrainG:
            batch_loss_g = pretrain_generator(g_net, train_data)
            # batch_loss_d = 0.0
            giter += 1 
            
        epoch_loss_g += batch_loss_g
        # epoch_loss_d += batch_loss_d
        num_batches += 1

        # write losses to tensorboard for plots
        if args.pretrainG:
            gwriter.add_scalar('train_g_loss', batch_loss_g,
                              giter)
        # elif args.pretrainD:
        #     dwriter.add_scalar('train_d_loss', batch_loss_d,
        #                       diter)
        # elif args.train:
        #     gwriter.add_scalar('train_g_loss', batch_loss_g,
        #                       giter)
        #     gwriter.add_scalar('mean_batch_rewward', mean_batch_reward, giter)
        #     dwriter.add_scalar('train_d_loss', batch_loss_d,
        #                       diter)
    epoch_loss_g = epoch_loss_g / num_batches
    # epoch_loss_d = epoch_loss_d / num_batches
        
    return epoch_loss_g, giter, gwriter
    

# ***** Inference for the GAN model ********
def val_model(g_net, data_generator, save_dir, epoch, vocab, 
              batchsize, writer, valiter=None):
    
    tot_val_loss = 0.0    
    num_batches = 0
    if args.eval != '':
        output_txt = open(os.path.join(
            #TODO save with timestamp
            save_dir, "generated_captions_e" + str(epoch) + ".txt"), "w+", 
            encoding='utf-8')
        output_json = os.path.join(save_dir, 
                                   "test_results_e" + str(epoch) + ".json")
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
                    gen_seq, gen_seq_logprob = g_net.sample(batch_image_feats, batch_paragraph)
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
        paragraph = model_utils.vec_2_text(paragraph, vocab['i2w'])
        print("TEXT: ", paragraph[0])

        gen_seq = gen_seq.cpu().numpy().tolist()
        gen_seq = model_utils.vec_2_text(gen_seq, vocab['i2w'])
        print("GENERATED: ", gen_seq[0])

        gt_abstract = batch_abstract.cpu().numpy()[:, 1:].tolist()
        gt_abstract = model_utils.vec_2_text(gt_abstract, vocab['i2w'])
        print("GROUND TRUTH: ", gt_abstract[0])

        output_txt.write(
                batch_images_file_names[0] + "\t" +  
                paragraph[0] +
                "\t" + gt_abstract[0] +
                "\t" + gen_seq[0] + "\n")

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
    
    # generate new vocab or load existing vocab
#     vocab, save_dir = init_vocab(os.path.join(data_dir, train_data_file))
    vocab, save_dir = init_vocab(os.path.join(data_dir, whole_data_file))
    
    # initalize training, val and test data generators
    train_generator, val_generator, test_generator = get_datagen(vocab)
    
    # get the models, optimizers, config, rollout and langeval initialized
    g_net,  g_optim = init_models(vocab) #d_net, lang_eval, rollout,

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
        # d_net, d_epoch, d_niter, d_val_epoch, d_val_iter = dis_stuff

    else:
        g_epoch = 1
        # d_epoch = 1
        g_val_epoch = 1
        # d_val_epoch = 1
        g_niter = 0
        # d_niter = 0
        g_val_iter = 0
        # d_val_iter = 0
    g_writer = SummaryWriter(save_dir)
    # d_writer = SummaryWriter(save_dir)
        
    print('saving directory: {}'.format(save_dir))
    
    if args.eval != '':
        print('inference mode')
        
        _,_ = val_model(g_net, test_generator, save_dir, g_epoch, vocab, 
                        config['dev_batch_size'], g_writer)

    else:
        if args.pretrainG:
            print("Pre-training generator network")
        # elif args.pretrainD:
        #     print("Pre-training discriminator network")
        # elif args.train:
        #     print("Jointly training generator and discriminator network")
        else:
            raise ValueError('none of the modes are ON. Check..')

        for epoch in range(1, int(args.epoch) + 1):
            print('==> Training:')
            print("Epoch = {}".format(epoch))
            
            # train model for 1 epoch
            (epoch_loss_g, g_niter, g_writer) = train_model(train_generator, g_net, #d_net,
                                               #rollout, lang_eval,
                                               g_epoch, #d_epoch,
                                               g_niter, #d_niter,
                                               g_writer, #d_writer,
                                               vocab)            
            # validate model every val_freq epochs
            # if epoch % args.val_freq == 0 and (args.pretrainG or args.train):
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

                model_utils._save_model(args, g_net, #d_net,
                                        g_optim,
                                        g_epoch, #d_epoch,
                                        g_val_epoch, #d_val_epoch ,
                                        g_niter, #d_niter,
                                        g_val_iter, #d_val_iter,
                                        save_dir)
            # update rollout if necessary
            # if args.train and args.use_rollout:
            #     rollout.update_params()
            
            # print loss
            print("GenEpoch = {}, Gen Loss = {}".format(g_epoch, epoch_loss_g))
            # print("DisEpoch = {}, Dis Loss = {}".format(d_epoch, epoch_loss_d))
            
            # log loss values and increase epoch number by 1
            if args.pretrainG:
                g_writer.add_scalar('train_g_epoch_loss', epoch_loss_g,
                                  g_epoch)
                g_epoch += 1
                
    g_writer.close()
    # d_writer.close()
            
    return save_dir, g_epoch #, d_epoch

if __name__ == '__main__':

    args = parse_args()

    if args.train_all:

        modargs = args
        # pretraining gen
        modargs.pretrainG = True
        modargs.pretrainD = False
        modargs.train = False
        modargs.epoch = 20
        savedir, g_epoch, d_epoch = main(modargs)
        
        # pretraining Dis
        if modargs.dis_type != 'none':
            modargs = args
            modargs.pretrainG = False
            modargs.pretrainD = True
            modargs.train = False
            modargs.epoch = 2
            modargs.resume = [savedir, 'gen_e{}.ckpt'.format(g_epoch - 1)]
            savedir, g_epoch, d_epoch = main(modargs)
        
        # training both
        modargs = args
        modargs.pretrainG = False
        modargs.pretrainD = False
        modargs.train = True
        modargs.epoch = 20
        modargs.resume = [savedir, 'gen_e{}.ckpt'.format(g_epoch - 1)]
        savedir, g_epoch, d_epoch = main(modargs)
    else:
        main(args)
