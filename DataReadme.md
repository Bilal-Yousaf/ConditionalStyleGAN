# Follow these Steps to run training from custom data

## Requirements
tensorflow-gpu 1.15.0 [No version later than this]
Commnad that can be used to install this: 
'''
pip install tensorflow-gpu==1.15.0
'''

## Dataset Structure
Dataset should have images related to classes placed in the respective folders (names of the folders should be classnames) and all these folders should be placed in a separate folder with none other file or folder placed in the root data folder.

## Clone the repository
```
git clone https://github.com/Bilal-Yousaf/ConditionalStyleGAN
cd ConditionalStyleGAN/
```

## Prepare Pickle file
```
python create_pickle.py -d <path of the root dataset folder>
For Example: python create_pickle.py -d '/content/sample_dataset'
```

## Convert Pickle file to TfRecords
```
python dataset_tool.py create_from_images dataset/logos '' 1
```

## Configuration Changes

### Step 1:
#### Number of class-conditions
<i>These lines refer to section at which you can also adjust hyperparameters.</i>
- ```./training/networks_stylegan.py``` line 388 & line 569:

> ``` label_size = 10```

### Step 2:
Set hyper-parameters for networks and other indications for the training loop

#### General
Starting at line 112 in ```training_loop.py```:
```
G_smoothing_kimg        = 10.0,     # Half-life of the running average of generator weights.
D_repeats               = 2,        # How many times the discriminator is trained per G iteration.
minibatch_repeats       = 1,        # Number of minibatches to run before adjusting training parameters.
reset_opt_for_new_lod   = True,     # Reset optimizer internal state (e.g. Adam moments) when new layers are introduced?
total_kimg              = 20000,    # Total length of the training, measured in thousands of real images.
mirror_augment          = True,     # Enable mirror augment?
drange_net              = [-1,1],   # Dynamic range used when feeding image data to the networks.
```
#### Mapping Network
Starting at line 384 in ```networks_stylegan.py```:
```
dlatent_size            = 128,          # Disentangled latent (W) dimensionality.
mapping_layers          = 8,            # Number of mapping layers.
mapping_fmaps           = 128,          # Number of activations in the mapping layers.
mapping_lrmul           = 0.01,         # Learning rate multiplier for the mapping layers.
mapping_nonlinearity    = 'lrelu',      # Activation function: 'relu', 'lrelu'.
use_wscale              = True,         # Enable equalized learning rate?
normalize_latents       = True,         # Normalize latent vectors (Z) before feeding them to the mapping layers?
```
#### Synthesis Network
Starting at line 384 in ```networks_stylegan.py```:
```
resolution          = 128,          # Output resolution.
fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
fmap_max            = 128,          # Maximum number of feature maps in any layer.
use_styles          = True,         # Enable style inputs?
const_input_layer   = True,         # First layer is a learned constant?
use_noise           = True,         # Enable noise inputs?
randomize_noise     = True,         # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu'
use_wscale          = True,         # Enable equalized learning rate?
use_pixel_norm      = False,        # Enable pixelwise feature vector normalization?
use_instance_norm   = True,         # Enable instance normalization?
dtype               = 'float32',    # Data type to use for activations and outputs.
fused_scale         = 'auto',       # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
blur_filter         = [1,2,1],      # Low-pass filter to apply when resampling activations. None = no filtering.

```
#### Discriminator Network

Starting at line 384 in ```networks_stylegan.py```:

```
fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
fmap_max            = 128,          # Maximum number of feature maps in any layer.
nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu',
use_wscale          = True,         # Enable equalized learning rate?
mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
fused_scale         = 'auto',       # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
blur_filter         = [1,2,1],      # Low-pass filter to apply when resampling activations. None = no filtering.
```

## Start Training
```
python train.py
```
