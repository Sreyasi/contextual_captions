# Contextual Captions

Modern web content – news articles, blog posts, educational resources, marketing brochures – is predominantly multimodal. A notable trait is the inclusion of media such as images placed at meaningful locations within a textual narrative. Most often, such images are accompanied by captions – either factual or stylistic (humorous, metaphorical, etc.) – making the narrative more engaging to the reader. While standalone image captioning has been extensively studied, captioning an
image based on external knowledge such as its surrounding text remains under-explored. In this work, we study this new task: given an image and an associated unstructured knowledge snippet, the goal is to generate a contextual caption for the image.

## Prerequisites

Python 3.6
PyTorch 1.0

## Configure

Update the following in the `config.json` file. 

"data_dir"  
"image_data_dir" 
"image_feat_data_dir"
"train_data_file"  
"test_data_file"   
"val_data_file"

## Dataset

Posts from the subreddit /r/pics are collected over a span of 1 year. Each post contains 1 image, its title (which we consider as image caption), and user comments. We concatenate up to 10 comments into a paragraph, and use it as additional knowledge or context for image captioning.

The dataset is partitiouned based on the presence of Named Entities (NE) in image captions. These partitions are called 'with_NE' and 'without_NE', and can be found in the 'data' folder.

A small set of test data can be found in data/test_small.json, and the corresponding image features are provided in image_feats/image_feats_test_small. The image features are generated from ResNet152.

## Evaluation

We use the MSCOCO automatic caption evaluation tool for evaluation. The correcponding codebase can be found here: https://github.com/tylin/coco-caption.

## Citation

```
@inproceedings{DBLP:conf/eacl/NagChowdhuryLANTERN21,
  author    = {Sreyasi {Nag Chowdhury} and
               Rajarshi Bhowmik and
               Harish Ravi and
			   Gerard de Melo and
			   Simon Razniewski and
               Gerhard Weikum},
  title     = {Exploiting Image-Text Synergy for Contextual Image Captioning},
  booktitle = {The Third Workshop Beyond Vision and LANguage: inTEgrating Real-world kNowledge. EACL Workshop LANTERN 2021}
}
```
