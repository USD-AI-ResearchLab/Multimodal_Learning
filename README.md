
# Multimodal Learning: Generating Precise Chest X-Ray Report on Thorax Abnormality 

## Appilied Artificial Intelligence Research Lab, University of South Dakota,
Vermillion, SD, 57069, USA
gaurab.subedi@coyotes.usd.edu, 
jayakumar.pujar@coyotes.usd.edu 

Department of Computer Science, University of South Dakota, Vermillion, SD,
57069, USA
santosh.kc@usd.edu


## Introduction

Chronic respiratory diseases, ranking as the third leading cause of death worldwide according to the 2017 World Health Organization (WHO) report, affect a staggering 544.9 million individuals. Compounding this public health challenge is the fact that over 80% of health systems grapple with shortages in their radiology departments, highlighting an urgent need for accessible and efficient diagnostic solutions. While various image classification models for analysing thorax abnormalities have been developed, relying solely on one type of dataset (image data, for example) for thorax abnormality analysis is insufficient. Integrating texts with image data could provide more accuracy as well as analysis. In response to this challenge, we propose a multimodal approach to generate detailed radiology reports from chest X-ray images and their corresponding radiological reports (Impression and Findings). Our framework integrates a pre-trained Convolutional Neural Network (CNN) for robust image feature extraction, a Recurrent Neural Network (RNN), and a visual attention mechanism to ensure coherent sentence generation. The image encoder employs the ResNet152 architecture to extract nuanced visual features from chest X-ray images. Simultaneously, the sentence generation model utilizes a Long Short-Term Memory (LSTM) layer to process textual data and generate contextually relevant reports. On an IU dataset of 7470 pairs of X-ray images and 3995 reports, our model exhibited superior performance based on language generation metrics (BLEU1= 0.4424, BLEU2= 0.2923, BLEU3= 0.207, BLEU4= 0.1464, ROUGE= 0.3396, and CIDEr= 0.2268), providing accurate and coherent impressions and findings compared to other benchmark models. 

##    Model Architecture: 

Encoder-decoder architecture where we used ResNet-152 CNN as an image encoder which extracts global and local visual features. LSTM is used as sentence decoder which generates sentences sequentially using image features and previous sentence where attention mechanism focuses on relevant image regions for each sentence. 

Training: Trained end to end using IU datasets. Optimized cross-entropy loss between generated and ground truth sentences. Teacher forcing is used for training and greedy search for testing. 

Evaluation Metrices: We used Natural Language Generation (NLG) Metrices like BLEU, ROGUE, CIDER. Higher scores are achieved when compared with different models which demonstrate their effectiveness. 

## Getting started

```
$ pip install -r requirements.txt
```
This Python script defines classes and functions for processing a JSON file, building a vocabulary from it, and saving that vocabulary as a pickle file.

```
$ cd IUdata
$ python3 vocab_creation.py
```
Convert XML file to JSON file for training

```
$ python3 convert_XML_to_json.py
```
Provide the dataset path and adjust hyperparameters in training file
```
$ cd ..
```
    #path
    parser.add_argument('--model_path', type=str, default='model_weights/', help='path for saving checkpoints')
    parser.add_argument('--vocab_path', type=str, default='IUdata/data_vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='IUdata/FrontalView_IUXRay', help='directory for X-ray images')
    parser.add_argument('--json_dir', type=str, default='IUdata/data_trainval.json', help='the path for json file')
    parser.add_argument('--log_path', default='./results', type=str, help='The path that stores the log files.')

    # model parameters
    parser.add_argument('--resize_size', type=int, default=256, help='The resize size of the X-ray image')
    parser.add_argument('--crop_size', type=int, default=224, help='The crop size of the X-ray image')

    # training parameters
    parser.add_argument('--teach_rate', type=float, default=1.0, help='The teach forcing rate in training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='The learning rate in training')
    parser.add_argument('--epochs', type=int, default=100, help='The epochs in training')
    parser.add_argument('--sche_step_size', type=int, default=5)
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='The dropout rate for both encoder and decoder')
    parser.add_argument('--eval_json_dir', type=str, default='IUdata/data_test.json', help='the path for json file')

```
$ python3 training.py
```
Similarly adjust the dataset path in testing file

```
$ python3 testing.py
```

## Dataset

Used Indiana University Chest X-Ray Collection which contains 7,471 chest x-ray images and 3,955 corresponding radiology reports.

Images:  https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz 

Reports: https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz  

