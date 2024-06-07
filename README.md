## CIADTI
we propose an end
to end deep learning framework to extract robust intermolecular and
intramolecular features of drugs and targets, and predict potential
DTIs, named CIADTI. Three parallel modules are proposed, that
one module is utilized to extract intermolecular features between
drugs and targets, while another two modules are used to produce
intramolecular features for drugs and targets, respectively. Finally,
all features are concatenated and fed into fully connected dense
layers for predicting DTIs. 

## Requirements
- python >= 3.5
- torch >= 1.4.0
- RDkit >= 2019.03.30
- gensim >= 3.4.0
- numpy >= 1.16.1
- pandas >= 1.1.4
- transformers >= 3.1.0

## Using
1. model.py: the construction of the neural network
2. data_preprocess: Process the data to get the input of the model
3. main.py: start file for model training

## Contact
if you have any questions or suggestions with the codes, please let us know, Contact Zhongjian Cheng at 448386695@qq.com
