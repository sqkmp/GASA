import argparse
from batchgenerators.utilities.file_and_folder_operations import *
import nnunet
import numpy as np
import SimpleITK as sitk
from multiprocessing import Pool
import nibabel as nib
import shutil
from collections import OrderedDict
import pkgutil
import importlib

from trainer.nnUNetTrainerV2 import nnUNetTrainerV2 as V2



network_training_output_dir = "/orange/unet_data/nnUNet_trained_models/nnunet_qkv50d25"
default_plans_identifier = "nnUNetPlansv2.1"
default_cascade_trainer = "nnUNetTrainerV2CascadeFullRes"
default_trainer = "nnUNetTrainerV2"
default_plans_identifier = "nnUNetPlansv2.1"



nnUNet_raw_data = "/orange/unet_data/nnUNet_raw_data_base/nnUNet_raw_data"
nnUNet_cropped_data = "/orange/unet_data/nnUNet_raw_data_base/nnUNet_cropped_data"
preprocessing_output_dir = "/orange/unet_data/nnUNet_preprocessed"

def recursive_find_python_class(folder, trainer_name, current_module):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, trainer_name):
                tr = getattr(m, trainer_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class([join(folder[0], modname)], trainer_name, current_module=next_current_module)
            if tr is not None:
                break

    return tr

def summarize_plans(file):
    plans = load_pickle(file)
    print("num_classes: ", plans['num_classes'])
    print("modalities: ", plans['modalities'])
    print("use_mask_for_norm", plans['use_mask_for_norm'])
    print("keep_only_largest_region", plans['keep_only_largest_region'])
    print("min_region_size_per_class", plans['min_region_size_per_class'])
    print("min_size_per_class", plans['min_size_per_class'])
    print("normalization_schemes", plans['normalization_schemes'])
    print("stages...\n")

    for i in range(len(plans['plans_per_stage'])):
        print("stage: ", i)
        print(plans['plans_per_stage'][i])
        print("")

def get_default_configuration(network, task, network_trainer, plans_identifier=default_plans_identifier,
                              search_in=("/home/ck/Renal_image/nnUNet/nnunet", "training", "network_training"),
                              base_module='nnunet.training.network_training'):
    assert network in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'], \
        "network can only be one of the following: \'2d\', \'3d_lowres\', \'3d_fullres\', \'3d_cascade_fullres\'"

    dataset_directory = join(preprocessing_output_dir, task)

    if network == '2d':
        plans_file = join(preprocessing_output_dir, task, plans_identifier + "_plans_2D.pkl")
    else:
        plans_file = join(preprocessing_output_dir, task, plans_identifier + "_plans_3D.pkl")

    plans = load_pickle(plans_file)
    possible_stages = list(plans['plans_per_stage'].keys())

    if (network == '3d_cascade_fullres' or network == "3d_lowres") and len(possible_stages) == 1:
        raise RuntimeError("3d_lowres/3d_cascade_fullres only applies if there is more than one stage. This task does "
                           "not require the cascade. Run 3d_fullres instead")

    if network == '2d' or network == "3d_lowres":
        stage = 0
    else:
        stage = possible_stages[-1]

    trainer_class = V2

    output_folder_name = join(network_training_output_dir, network, task, network_trainer + "__" + plans_identifier)

    print("###############################################")
    print("I am running the following nnUNet: %s" % network)
    print("My trainer class is: ", trainer_class)
    print("For that I will be using the following configuration:")
    summarize_plans(plans_file)
    print("I am using stage %d from these plans" % stage)

    if (network == '2d' or len(possible_stages) > 1) and not network == '3d_lowres':
        batch_dice = True
        print("I am using batch dice + CE loss")
    else:
        batch_dice = False
        print("I am using sample dice + CE loss")

    print("\nI am using data from this folder: ", join(dataset_directory, plans['data_identifier']))
    print("###############################################")
    return plans_file, output_folder_name, dataset_directory, batch_dice, stage, trainer_class

def convert_id_to_task_name(task_id: int):
    startswith = "Task%03.0d" % task_id
    if preprocessing_output_dir is not None:
        candidates_preprocessed = subdirs(preprocessing_output_dir, prefix=startswith, join=False)
    else:
        candidates_preprocessed = []

    if nnUNet_raw_data is not None:
        candidates_raw = subdirs(nnUNet_raw_data, prefix=startswith, join=False)
    else:
        candidates_raw = []

    if nnUNet_cropped_data is not None:
        candidates_cropped = subdirs(nnUNet_cropped_data, prefix=startswith, join=False)
    else:
        candidates_cropped = []
    
    candidates_trained_models = []
    
    print(network_training_output_dir)
    if network_training_output_dir is not None:
        for m in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres']:
            if isdir(join(network_training_output_dir, m)):
                candidates_trained_models += subdirs(join(network_training_output_dir, m), prefix=startswith, join=False)

    all_candidates = candidates_cropped + candidates_preprocessed + candidates_raw + candidates_trained_models
    unique_candidates = np.unique(all_candidates)
    if len(unique_candidates) > 1:
        raise RuntimeError("More than one task name found for task id %d. Please correct that. (I looked in the "
                           "following folders:\n%s\n%s\n%s" % (task_id, nnUNet_raw_data, preprocessing_output_dir,
                                                               nnUNet_cropped_data))
    if len(unique_candidates) == 0:
        raise RuntimeError("Could not find a task with the ID %d. Make sure the requested task ID exists and that "
                           "nnU-Net knows where raw and preprocessed data are located (see Documentation - "
                           "Installation). Here are your currently defined folders:\nnnUNet_preprocessed=%s\nRESULTS_"
                           "FOLDER=%s\nnnUNet_raw_data_base=%s\nIf something is not right, adapt your environemnt "
                           "variables." %
                           (task_id,
                            os.environ.get('nnUNet_preprocessed') if os.environ.get('nnUNet_preprocessed') is not None else 'None',
                            os.environ.get('RESULTS_FOLDER') if os.environ.get('RESULTS_FOLDER') is not None else 'None',
                            os.environ.get('nnUNet_raw_data_base') if os.environ.get('nnUNet_raw_data_base') is not None else 'None',
                            ))
    return unique_candidates[0]








def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")
    parser.add_argument("task", help="can be task name or task id")
    parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'')
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                             "is much more CPU and RAM intensive and should only be used if you know what you are "
                             "doing", required=False)
    parser.add_argument("--deterministic",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                             "this is not necessary. Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the validation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--find_lr", required=False, default=False, action="store_true",
                        help="not used here, just for fun")
    parser.add_argument("--valbest", required=False, default=False, action="store_true",
                        help="hands off. This is not intended to be used")
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")
    parser.add_argument("--val_folder", required=False, default="validation_raw",
                        help="name of the validation folder. No need to use this for most people")
    parser.add_argument("--disable_saving", required=False, action='store_true',
                        help="If set nnU-Net will not save any parameter files (except a temporary checkpoint that "
                             "will be removed at the end of the training). Useful for development when you are "
                             "only interested in the results and want to save some disk space")
    parser.add_argument("--disable_postprocessing_on_folds", required=False, action='store_true',
                        help="Running postprocessing on each fold only makes sense when developing with nnU-Net and "
                             "closely observing the model performance on specific configurations. You do not need it "
                             "when applying nnU-Net because the postprocessing for this will be determined only once "
                             "all five folds have been trained and nnUNet_find_best_configuration is called. Usually "
                             "running postprocessing on each fold is computationally cheap, but some users have "
                             "reported issues with very large images. If your images are large (>600x600x600 voxels) "
                             "you should consider setting this flag.")
    # parser.add_argument("--interp_order", required=False, default=3, type=int,
    #                     help="order of interpolation for segmentations. Testing purpose only. Hands off")
    # parser.add_argument("--interp_order_z", required=False, default=0, type=int,
    #                     help="order of interpolation along z if z is resampled separately. Testing purpose only. "
    #                          "Hands off")
    # parser.add_argument("--force_separate_z", required=False, default="None", type=str,
    #                     help="force_separate_z resampling. Can be None, True or False. Testing purpose only. Hands off")
    parser.add_argument('--val_disable_overwrite', action='store_false', default=True,
                        help='Validation does not overwrite existing segmentations')
    parser.add_argument('--disable_next_stage_pred', action='store_true', default=False,
                        help='do not predict next stage')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='path to nnU-Net checkpoint file to be used as pretrained model (use .model '
                             'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')

    args = parser.parse_args()

    task = args.task
    fold = args.fold
    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only
    plans_identifier = args.p
    find_lr = args.find_lr
    disable_postprocessing_on_folds = args.disable_postprocessing_on_folds

    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data

    deterministic = args.deterministic
    valbest = args.valbest

    fp32 = args.fp32
    run_mixed_precision = not fp32

    val_folder = args.val_folder
    # interp_order = args.interp_order
    # interp_order_z = args.interp_order_z
    # force_separate_z = args.force_separate_z

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    # if force_separate_z == "None":
    #     force_separate_z = None
    # elif force_separate_z == "False":
    #     force_separate_z = False
    # elif force_separate_z == "True":
    #     force_separate_z = True
    # else:
    #     raise ValueError("force_separate_z must be None, True or False. Given: %s" % force_separate_z)
    print(network)
    print(task)
    print(network_trainer)
    print(plans_identifier)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
    trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)
    print("###################################################################################################")
    print(plans_file)
    print(output_folder_name)
    print(dataset_directory)
    print(batch_dice)
    print(stage)
    print(trainer_class)
    print("###################################################################################################")


    trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                            deterministic=deterministic,
                            fp16=run_mixed_precision)

    if args.disable_saving:
        trainer.save_final_checkpoint = False # whether or not to save the final checkpoint
        trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to
        # self.best_val_eval_criterion_MA
        trainer.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest. We need that in case
        # the training chashes
        trainer.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each

    trainer.initialize(not validation_only)

    
    if args.disable_saving:
        trainer.save_final_checkpoint = False # whether or not to save the final checkpoint
        trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to
        # self.best_val_eval_criterion_MA
        trainer.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest. We need that in case
        # the training chashes
        trainer.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each

    trainer.initialize(not validation_only)
    if find_lr:
        trainer.find_lr()
    else:
        if not validation_only:
            if args.continue_training:
                # -c was set, continue a previous training and ignore pretrained weights
                trainer.load_latest_checkpoint()
            elif (not args.continue_training) and (args.pretrained_weights is not None):
                # we start a new training. If pretrained_weights are set, use them
                load_pretrained_weights(trainer.network, args.pretrained_weights)
            else:
                # new training without pretraine weights, do nothing
                pass

            trainer.run_training()
        else:
            if valbest:
                trainer.load_best_checkpoint(train=False)
            else:
                trainer.load_final_checkpoint(train=False)

        trainer.network.eval()

        # predict validation
        trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder,
                         run_postprocessing_on_folds=not disable_postprocessing_on_folds,
                         overwrite=args.val_disable_overwrite)

        if network == '3d_lowres' and not args.disable_next_stage_pred:
            print("predicting segmentations for the next stage of the cascade")
            predict_next_stage(trainer, join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1))



if __name__ == "__main__":
    main()