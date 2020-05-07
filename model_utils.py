# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:49:06 2019

@author: HareeshRavi
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import torch
import os

# convert sequence of tokens to seq of words
def vec_2_text(vecdata, idx_to_w):       
    '''
    converts sequence of tokens to actual sentence
    Args:
        vecdata : (batchsize, seqlen) of word tokens
    Returns:
        textdata: (batchsize, seqlen) of words
    '''
    # get Gt sentence from input vecs
    textdata = []
    for i in range(len(vecdata)):
        gt_abstract = []
        for w_idx in vecdata[i]:
            try:
                w_idx = w_idx.item()
            except AttributeError:
                pass
            if w_idx == 2: # Ignore everyting after <eos>
                break
            gt_abstract.append(idx_to_w[w_idx])
        gt_abstract = " ".join(gt_abstract)
        textdata.append(gt_abstract)
    return textdata

def text_2_vec(textdata, w_to_idx):
    '''
    Converts text to sequence of tokens
    :param:textdata: (batchsize , variable sequence len)
    :param w_to_idx:
    :return:
    '''
    vecdata = []
    for i in range(len(textdata)):
        vec = []
        for w in textdata[i].split(' '):
            if w in w_to_idx:
                vec.append(w_to_idx[w])
        vecdata.append(vec)
    return vecdata

# create labels for dis after smoothing
def randsel(labeltype):
    '''
    selects a random value in [0.7, 1.0] for 1 and [0.0, 0.3] for 0
    Args:
        labeltype: (1) True / False (for real/generated captions)
    Returns:
        (1) value randomly selected between corresponding ranges for True/False
    '''
    val = np.random.rand(1)
    if labeltype:                
        return (val * 0.3) + 0.7
    else:
        return (val * 0.3) + 0.0

# write seq of tokens to file    
def write_samples_to_file(samples, filename):
    
    with open(filename, 'w') as f:
        for sample in samples:
            cur_sent = []
            for s in sample:
                if s == 2:
                    cur_sent.append(str(2))
                    break
                else:
                    cur_sent.append(str(s))
                
            string = ' '.join(cur_sent)
            f.write('{}\n'.format(string))
    return True
            
# read seq of tokens from file
def read_samples_from_file(file):
    '''
    Reading sequence of generated tokens for a batch
    Args:
        file: (str) filename to be read
        
    Returns:
        data: (batchsize, seqlen) tokens in the file
    '''
    with open(file, 'r') as f:
        lines = f.readlines()
    data = []
    data_len = []
    for line in lines:
        l = line.strip().split(' ')
        l = [int(s) for s in l]
        data.append(l)
        data_len.append(len(l))
    return data, data_len

def write_dis_input_to_file(flag, epoch, 
                            real_text, gt_real_probs, 
                            gen_text, gen_real_probs, 
                            unpaired_text=None, unpaired_real_probs=None):
    
    # Print GT and gen sent that are fed into discriminator and the out prob
    if flag.pretrainD:
        output_file_dis_input = open("./discriminator_inputs_pretrain.txt", 
                                     "a+", encoding='utf-8')
    elif flag.train:
        output_file_dis_input = open("./discriminator_inputs_train.txt", 
                                     "a+", encoding='utf-8')
    else:
        raise ValueError('this shouldnt be called otherwise..')
    
    output_file_dis_input.write("Epoch " + str(epoch) + "\n")
    sampleidx = random.sample(range(len(real_text)), 2)
    
    output_file_dis_input.write("GROUND TRUTHs:\n")
    for i in sampleidx:
        output_file_dis_input.write("Sent: {} \n Prob: {}\n".format(
                real_text[i],
                gt_real_probs.tolist()[i]))
        output_file_dis_input.write("\n_______________________________" + 
                                    "__________________________\n")
        output_file_dis_input.flush()
    
    output_file_dis_input.write("GENERATED:\n")
    for i in sampleidx:
        output_file_dis_input.write("Sent: {} \n Prob: {}\n".format(
                gen_text[i], 
                gen_real_probs.tolist()[i]))
        output_file_dis_input.write("\n_______________________________" + 
                                    "__________________________\n")
        output_file_dis_input.flush()
    
    if unpaired_text is not None:
        output_file_dis_input.write("UNPAIRED:\n")
        for i in sampleidx:
            output_file_dis_input.write("Sent: {} \n Prob: {}\n".format(
                    unpaired_text[i], 
                    unpaired_real_probs.tolist()[i]))
            output_file_dis_input.write("\n_______________________________" + 
                                        "__________________________\n")
            output_file_dis_input.flush()
    return True

def write_gen_output_to_file(epoch, imagefiles, real_text, gen_text, 
                             final_rewards):
    
    output_file_dis_input = open("./generator_outputs_train.txt", 
                                 "a+", encoding='utf-8')
    output_file_dis_input.write("Epoch " + str(epoch) + "\n")
    sampleidx = random.sample(range(len(gen_text)), 2)
    output_file_dis_input.write("GENERATED:\n")
    for i in sampleidx:
        output_file_dis_input.write(
                "Image: {} \nReal Sent: {} \nGen Sent: {} \nReward to Gen: {}\n".format(
                imagefiles[i], real_text[i], gen_text[i], final_rewards[i][0]))
    output_file_dis_input.write("\n_______________________________" + 
                                "__________________________\n")
    return True

# truncate gen samples until EOS and return along with lengths
def process_gen_sentences(gen_sent):
    
    prepro_samples = []
    prepro_lengths = []
    for sentence in gen_sent:
        prepro_sentence = []
        for word in sentence:
            if word != 2:
                prepro_sentence.append(word)
            else:
                # to keep parity with the ground truth, ground truth 
                # sentences end with <eos>, so generated sentence should 
                # also end with <eos>
                prepro_sentence.append(word) 
                break
        # handles empty string generation issue
        if len(prepro_sentence) < 1: 
            prepro_sentence.append(0)
        prepro_samples.append(prepro_sentence)
        prepro_lengths.append(len(prepro_sentence))
    return prepro_samples, prepro_lengths

# load checkpoints
def _load_model(args, model, config):
    
    if args.eval != '':
        genpath = args.eval
        save_dir = '/'.join(genpath.split('/')[:-1])
    elif args.resume != '':
        save_dir = args.resume[0]
        dispath = None
        if len(args.resume) > 1:
            genpath = os.path.join(save_dir, args.resume[1])
        if len(args.resume) > 2:
            dispath = os.path.join(save_dir, args.resume[2])
        if len(args.resume) < 2 or len(args.resume) > 3:
            raise ValueError('Too many or too little args --resume')
    else:
        raise ValueError('No eval or resume file path to load from..')
    
    # get initialized models
    g_net = model #, d_net
    
    print('load Generator from ckpt:', genpath)
    ckpt_gen = torch.load(genpath)

    # load already saved states to the initialized models
    g_net.load_state_dict(ckpt_gen['generator'])
#    if not args.train:
#     g_net.set_optim(config['lr'], config['optimizer'],
#                     ckpt_gen['gen_optimizer'])
#    else:
#        g_net.set_optim(config['lr'], config['optimizer'])
    g_epoch = ckpt_gen['epoch'] + 1
    g_val_epoch = ckpt_gen['valepoch']
    g_iter = ckpt_gen['niter'] + 1
    g_valiter = ckpt_gen['nvaliter'] + 1
    
    genobj = (g_net, g_epoch, g_iter, g_val_epoch, g_valiter)
    # save arguments during resuming
    
    argspath = os.path.join(save_dir, 'gen_args-' + str(g_epoch) + '.json')
    with open(argspath, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

        
    # if (args.pretrainD or args.train) and dispath is not None:
    #     print('load Discriminator from ckpt:', dispath)
    #     ckpt_dis = torch.load(dispath)
    #
    #     # load already saved states to the initialized models
    #     d_net.load_state_dict(ckpt_dis['discriminator'])
    #     if not args.train:
    #         d_net.set_optim(config['lr'], config['optimizer'],
    #                         ckpt_dis['dis_optimizer'])
    #     d_epoch = ckpt_dis['epoch'] + 1
    #     d_val_epoch = ckpt_dis['valepoch']
    #     d_iter = ckpt_dis['niter'] + 1
    #     d_valiter = ckpt_dis['nvaliter'] + 1
    #
    #     disobj = (d_net, d_epoch, d_iter, d_val_epoch, d_valiter)
    #     # save arguments during resuming
    #     argspath = os.path.join(save_dir, 'dis_args-' + str(d_epoch) + '.json')
    #     with open(argspath, 'w') as f:
    #         json.dump(args.__dict__, f, indent=2)
    # else:
    #     disobj = (d_net, 1, 0, 1, 0)
        
    
   
    return genobj, save_dir # disobj, g_writer, d_writer

# Save checkpoints
def _save_model(args, g_net, g_optim, g_epoch,
                g_val_epoch, g_iter,
                g_val_iter, savedir):#gwriter, dwriter, savedir):
    
    if args.pretrainG or args.train:
        # saving generator
        path = os.path.join(savedir, 'gen_e{}.ckpt'.format(g_epoch))
        print('save Generator checkpoint:', path)
    
        ckpt = {
                'generator': g_net.state_dict(),
                'gen_optimizer': g_optim.state_dict(),
                'epoch': g_epoch,
                'valepoch': g_val_epoch,
                'niter': g_iter,
                'nvaliter': g_val_iter
#                'writer': gwriter
                }        
        torch.save(ckpt, path)
    
#     if (args.pretrainD or args.train) and args.dis_type != 'none':
#         # saving Discriminator
#         path = os.path.join(savedir, 'dis_e{}.ckpt'.format(d_epoch))
#         print('save Discriminator checkpoint:', path)
#
#         ckpt = {
#                 'discriminator': d_net.state_dict(),
#                 'dis_optimizer': d_net.optimizer.state_dict(),
#                 'epoch': d_epoch,
#                 'valepoch': d_val_epoch,
#                 'niter': d_iter,
#                 'nvaliter': d_val_iter
# #                'writer': dwriter
#                 }
#         torch.save(ckpt, path)
    return True

# know whether gradient is present or not
def plot_grad_flow(named_parameters):
    '''
    This is used for checking the gradient flow
    Args:
        named_parameters: () parameters of gen or dis whose grad status will
                             be printed
    Returns:
        None
    '''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if p.grad is None:
                print(n, p.grad)
            else:
                print(n, "gradient is present")
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    return None

def convert_to_json(sentences):
    json_sent = {}
    for i, sent in enumerate(sentences):
        json_sent[i] = [sent]
    return json_sent

def get_lang_eval_scores(lang_eval, ref_sent, gen_sent):
    '''
    :param lang_eval: object
    :param ref_sent: reference sentences in json format
    :param gen_sent: hypothesis sentences in json format
    :return: SPIDEr scores
    '''
    scores = lang_eval.lang_scorer(ref_sent, gen_sent)
    return scores

# def write_ref(image_id, ref_sent):
#     data = {}
#     data["images"] = []
#     data["images"].append(
#         {
#         "id": image_id,
#         "file_name": image_id + ".jpg"
#         }
#     )
#     data["annotations"].append(
#         {
#         "id": image_id,
#         "image_id": image_id,
#         "caption": ref_sent
#         }
#     )
#     with open('ref.json', 'w+') as f:
#         json.dump(data, f)
#
# def write_hyp(image_id, gen_sent):
#     data = [
#         {
#         "image_id": image_id,
#         "caption": gen_sent
#         }
#     ]
#     with open('hyp.json', 'w+') as f:
#         json.dump(data, f)

# def get_lang_eval_scores(batch_image_id, batch_ref_sent, batch_gen_sent):
#     lang_eval_scores = []
#     for image_id, ref_sent, gen_sent in zip(batch_image_id, batch_ref_sent, batch_gen_sent):
#         write_ref(image_id, ref_sent)
#         write_hyp(image_id, gen_sent)
#         score = lang_scorer()
#         lang_eval_scores.append(score)
#     return lang_eval_scores

## ***** Saving model ********
#def save_model(args, savedir, g_net, d_net, epoch, final=False):
#
#    # is this final epoch or not
#    if final:
#        savedir = ''
#        epochnum = ''
#    else:
#        epochnum = '-e{}'.format(epoch)
#    
#    # actually save        
#    if args.pretrainG:
#        print("Pre-training generator network")
#        path = os.path.join(savedir, 
#                            './pretrained-generator{}.pth'.format(epochnum))
#        torch.save(g_net.state_dict(), path)
#        
#    elif args.pretrainD:
#        print("Pre-training discriminator network")
#        path = os.path.join(savedir, 
#                            './pretrained-discriminator{}.pth'.format(epochnum))
#        torch.save(d_net.state_dict(), path)
#
#    elif args.train:
#        print("Training generator and discriminator network")
#        path = os.path.join(savedir, './generator{}.pth'.format(epochnum))
#        torch.save(g_net.state_dict(), path)
#        path = os.path.join(savedir, './discriminator{}.pth'.format(epochnum))
#        torch.save(d_net.state_dict(), path)
#    else:
#        raise ValueError('train, pretrainG or pretrainD should be True')
#
#    return True
