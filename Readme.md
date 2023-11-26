# GASA-Unet
Our code is based on nnUNet.

# Enviroment
pip install requirements.txt

# Dataset creating
### python create_dataset.py

Don’t forget to change the direction:
##### base = "/data/datasets/kits23/dataset"  
##### out = "/data/datasets/kits23/nnunet_kits23".

Note: base for original dataset direction, covering images and labels for training and test, respectively.

# Data preprocessing

### python preprocessing.py -t ID --verify_dataste_integrity

Note: keep the default name of trainer

##### default_plans_identifier = "nnUNetPlansv2.1"
##### default_trainer = "nnUNetTrainerV2"

change the output direction 

##### preprocessing_output_dir = "/orange/unet_data/nnUNet_preprocessed"
##### network_training_output_dir = "/orange/unet_data/nnUNet_trained_models/nnUNet"
##### nnUNet_raw_data = "/orange/unet_data/nnUNet_raw_data_base/nnUNet_raw_data"
##### nnUNet_cropped_data = "/orange/unet_data/nnUNet_raw_data_base/nnUNet_cropped_data"

ID for the task ID, which contains 3 numbers.

# Training

### python train.py 3d_fullres nnUNetTrainerV2 ID Fold --fp32 --deterministic -c

Note:
ID : task ID. Fold : number of 0,1,2,3,4 for five-cross validation.

keep the default name of trainer 
##### default_plans_identifier = "nnUNetPlansv2.1"
##### default_cascade_trainer = "nnUNetTrainerV2CascadeFullRes"
##### default_trainer = "nnUNetTrainerV2"
##### default_plans_identifier = "nnUNetPlansv2.1"

keep the preprocessing output folder
##### nnUNet_raw_data = "/orange/unet_data/nnUNet_raw_data_base/nnUNet_raw_data"
##### nnUNet_cropped_data = "/orange/unet_data/nnUNet_raw_data_base/nnUNet_cropped_data"
##### preprocessing_output_dir = "/orange/unet_data/nnUNet_preprocessed"

change the output folder for trained model.
##### network_training_output_dir = "/orange/unet_data/nnUNet_trained_models/nnunet_qkv50d25"

##### Inference

### python inference.py -i input_folder -o output_folder -t ID -m 3d_fullres --disable_mixed_precision

keep the default name of trainer.
##### default_plans_identifier = "nnUNetPlansv2.1"
##### default_cascade_trainer = "nnUNetTrainerV2CascadeFullRes"
##### default_trainer = "nnUNetTrainerV2"

change the model output folder
##### network_training_output_dir = "/orange/unet_data/nnUNet_trained_models/nnunet_qkv50d1"

