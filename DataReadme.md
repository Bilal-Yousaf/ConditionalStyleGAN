# Follow these Steps to run training from custom data

### Requirements
tensorflow-gpu 1.15.0 [No version later than this]
Commnad that can be used to install this: 
'''
pip install tensorflow-gpu==1.15.0
'''

### Dataset Structure
Dataset should have images related to classes placed in the respective folders (names of the folders should be classnames) and all these folders should be placed in a separate folder with none other file or folder placed in the root data folder.

### Clone the repository
```
git clone https://github.com/Bilal-Yousaf/ConditionalStyleGAN
cd ConditionalStyleGAN/
```

### Prepare Pickle file
```
python create_pickle.py -d <path of the root dataset folder>
For Example: python create_pickle.py -d '/content/sample_dataset'
```

### Convert Pickle file to TfRecords
```
python dataset_tool.py create_from_images dataset/logos '' 1
```

### Start Training
```
python train.py
```
