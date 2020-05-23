from coco_caption.pycocoevalcap.cider.cider import Cider
from coco_caption.pycocoevalcap.spice.spice import Spice
from coco_caption.pycocoevalcap.bleu.bleu import Bleu
from coco_caption.pycocoevalcap.rouge.rouge import Rouge
from coco_caption.pycocoevalcap.meteor.meteor import Meteor

import json
import os
import numpy as np
from params import parse_args


class AutoEval():
    def __init__(self, ref_file):
        self.ref_corpus = []
        self.init_lang_eval(ref_file)
        self.cider = Cider(self.ref_corpus)
        self.spice = Spice()
        self.bleu = Bleu()
        self.rouge = Rouge()
        self.meteor = Meteor()

    def init_lang_eval(self, ref_file):
        with open(ref_file) as f:
            data = json.load(f)
            for sample in data:
                self.ref_corpus.append(sample["caption"])

    def lang_scorer(self, sent_gt, sent_gen):
        scores = {}


        bleu_avg_score, bleu_scores = self.bleu.compute_score(sent_gt, sent_gen)
#         print(bleu_avg_score[2])
        scores['BLEU-1'] = bleu_avg_score[0] * 100
        scores['BLEU-2'] = bleu_avg_score[1] * 100
        scores['BLEU-3'] = bleu_avg_score[2] * 100
        scores['BLEU-4'] = bleu_avg_score[3] * 100

#         meteor_avg_score, meteor_scores = self.meteor.compute_score(sent_gt, sent_gen)
#         print(meteor_avg_score)
#         scores['METEOR'] = meteor_avg_score * 100

        rouge_avg_score, rouge_scores = self.rouge.compute_score(sent_gt, sent_gen)
        scores['ROUGE-L'] = rouge_avg_score * 100

        cider_avg_score, cider_scores = self.cider.compute_score(sent_gt, sent_gen)
        scores['CIDEr'] = cider_avg_score * 100 # We didn't multiply by 10 inside cider compute_score.
        # So, to make it comparable with the Shuster et al., we multiply it by 100.
        # Shuster et al. used the CIDEr in COCO caption which was already multiplied by 10.
        # We speculate that they multipled it again by 10 to get their results.

        spice_avg_score, spice_scores = self.spice.compute_score(sent_gt, sent_gen)
        scores['SPICE'] = spice_avg_score * 10
        print('BLEU-1: ', str(round(scores['BLEU-1'], 2)))
        print('BLEU-2: ', str(round(scores['BLEU-2'], 2)))
        print('BLEU-3: ', str(round(scores['BLEU-3'], 2)))
        print('BLEU-4: ', str(round(scores['BLEU-4'], 2)))
        print('ROUGE-L: ', str(round(scores['ROUGE-L'], 2)))
        print('CIDEr: ', str(round(scores['CIDEr'], 2)))
        print('SPICE: ', str(round(scores['SPICE'], 2)))

        return scores

def main(args):

    # read the config file
    config_file = 'config.json'
    config = json.load(open(config_file, 'r'))
    data_dir = config['data_dir']
    image_data_dir = config['image_data_dir']
    test_data_file = config['test_data_file']

    # Initilize the language Evaluator
    ref_file = os.path.join(data_dir, test_data_file)
    lang_eval = AutoEval(ref_file)

    # Read the results file
    sent_gt = {}
    sent_gen = {}
    #result_file = '/GW/multimodal_embeddings/static00/reddit/praw/trained_models/run104653_2020_05_15_11_41_23_without_ne_80000/generated_captions_e21.txt'  #TODO: assign timestamp to file
    result_file = args.result_file
    with open(result_file, 'r') as f:
        idx = 0
        for line in f:
            try:
                _, _, gt, gen =line.strip().split('\t')
                sent_gt[idx] = [gt]
                sent_gen[idx] = [gen]
                idx += 1
            except:
                continue

    # Obtain the scores

    scores = lang_eval.lang_scorer(sent_gt, sent_gen)
    #print(scores)

if __name__ == '__main__':
    args = parse_args()
    main(args)