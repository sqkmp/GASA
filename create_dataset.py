

from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from scipy.ndimage import label


if __name__ == "__main__":
    base = "/data/datasets/kits23/dataset"
    out = "/data/datasets/kits23/nnunet_kits23"
    cases = subdirs(base, join=False)                       ### Read all subfolders, each of which contains one instance
    
    #######################################################################################################
    # Create folders for labels, trainning dataset and test dataset.
    maybe_mkdir_p(out)
    maybe_mkdir_p(join(out, "imagesTr"))
    maybe_mkdir_p(join(out, "imagesTs"))
    maybe_mkdir_p(join(out, "labelsTr"))
    maybe_mkdir_p(join(out, "labelsTs"))
    #######################################################################################################

    #######################################################################################################
    for c in cases:
        case_id = int(c.split("_")[-1])
        if case_id < 300:
            shutil.copy(join(base, c, "imaging.nii.gz"), join(out, "imagesTr", c + "_0000.nii.gz"))
            shutil.copy(join(base, c, "segmentation.nii.gz"), join(out, "labelsTr", c + ".nii.gz"))
        else:
            shutil.copy(join(base, c, "imaging.nii.gz"), join(out, "imagesTs", c + "_0000.nii.gz"))
            shutil.copy(join(base, c, "segmentation.nii.gz"), join(out, "labelsTs", c + ".nii.gz"))    
    #######################################################################################################
    c_train = cases[0:300]
    c_test = cases[300:]
    #######################################################################################################
    

    
    # json file
    json_dict = {}
    json_dict['name'] = "KiTS"
    json_dict['description'] = "kidney and kidney tumor segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "KiTS data for nnunet"
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "Kidney",
        "2": "Tumor",
        "3": "cyst"
    }
    json_dict['numTraining'] = 300
    json_dict['numTest'] = 189
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             c_train]
    json_dict['test'] = [{'image': "./imagesTs/%s.nii.gz" % i, "label": "./labelsTs/%s.nii.gz" % i} for i in
                             c_test]

    save_json(json_dict, os.path.join(out, "dataset.json"))
    #######################################################################################################