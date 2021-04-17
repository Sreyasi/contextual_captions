# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:07:23 2019

@author: HareeshRavi
"""
import argparse
import pdb


def parse_args():
    parser = argparse.ArgumentParser()

    # general parameters
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
            '--train', action='store_true', default=False,
            help='Switch this flag on to train the network')
    mode_group.add_argument(
            '--eval', default='', type=str,
            help='Give model file to be evaluated')

    # generator decoder specific parameters
    decoder_type = parser.add_mutually_exclusive_group()
    decoder_type.add_argument(
            '--greedy', action='store_true', default=False,
            help='Switch this flag on for greedy MLE decoding')
    decoder_type.add_argument(
            '--beam', action='store_true', default=False,
            help='Switch this flag on for beam serach decoding')

    # training parameters
    parser.add_argument('--preprocess', action='store_true', default=False,
                        help='Switch this flag on for creating new vocabulary')
    parser.add_argument('--epoch', help='Number of epochs', type=int)
    parser.add_argument('--val_freq', default=1, type=int,
                        help='Number of epochs between validation')
    parser.add_argument('--save_freq', default=2, type=int,
                        help='Number of epochs between saving model')
    parser.add_argument(
            '--resume', nargs='+', default='', type=str,
            help='Give model dir, gen epoch and dis epoch')
    parser.add_argument('--grad_clip', type=float, default=0.1,
                        help='clip gradients at this value')

    # Tokenizer
    parser.add_argument('--use_bert_tokenizer', action='store_true', default=False,
                        help='Switch this flag on to use bert tokenizer')

    # directories and file paths parameters
    parser.add_argument('--savedir', default='./trained_models/', type=str,
                        help='directory to save model checkpoints')
    parser.add_argument('--config_file', default='./config.json', type=str,
                        help='path to config')
    parser.add_argument('--result_file', default='', type=str,
                        help='path to result file for evaluation')

    FLAG = parser.parse_args()
    assert FLAG.greedy or FLAG.beam
    return FLAG


if __name__ == "__main__":
    parse_args()
