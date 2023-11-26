#from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir, nnUNet_cropped_data, network_training_output_dir
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from nnunet.configuration import default_num_threads
import SimpleITK as sitk
from multiprocessing import Pool
import nibabel as nib
import shutil
from collections import OrderedDict
import pkgutil
import importlib
import nnunet
from experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
import sys

import time

default_plans_identifier = "nnUNetPlansv2.1"
default_cascade_trainer = "nnUNetTrainerV2CascadeFullRes"
default_trainer = "nnUNetTrainerV2"
preprocessing_output_dir = "/orange/unet_data/nnUNet_preprocessed"
network_training_output_dir = "/orange/unet_data/nnUNet_trained_models/nnUNet"
default_plans_identifier = "nnUNetPlansv2.1"
nnUNet_raw_data = "/orange/unet_data/nnUNet_raw_data_base/nnUNet_raw_data"
nnUNet_cropped_data = "/orange/unet_data/nnUNet_raw_data_base/nnUNet_cropped_data"

class DatasetAnalyzer(object):
    def __init__(self, folder_with_cropped_data, overwrite=True, num_processes=default_num_threads):
        """
        :param folder_with_cropped_data:
        :param overwrite: If True then precomputed values will not be used and instead recomputed from the data.
        False will allow loading of precomputed values. This may be dangerous though if some of the code of this class
        was changed, therefore the default is True.
        """
        self.num_processes = num_processes
        self.overwrite = overwrite
        self.folder_with_cropped_data = folder_with_cropped_data
        self.sizes = self.spacings = None
        self.patient_identifiers = get_patient_identifiers_from_cropped_files(self.folder_with_cropped_data)
        #all cases' name
        assert isfile(join(self.folder_with_cropped_data, "dataset.json")), \
            "dataset.json needs to be in folder_with_cropped_data"
        self.props_per_case_file = join(self.folder_with_cropped_data, "props_per_case.pkl")
        self.intensityproperties_file = join(self.folder_with_cropped_data, "intensityproperties.pkl")

    def load_properties_of_cropped(self, case_identifier):
        with open(join(self.folder_with_cropped_data, "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return properties

    @staticmethod
    def _check_if_all_in_one_region(seg, regions):
        res = OrderedDict()
        for r in regions:
            new_seg = np.zeros(seg.shape)
            for c in r:
                new_seg[seg == c] = 1
            labelmap, numlabels = label(new_seg, return_num=True)
            if numlabels != 1:
                res[tuple(r)] = False
            else:
                res[tuple(r)] = True
        return res

    @staticmethod
    def _collect_class_and_region_sizes(seg, all_classes, vol_per_voxel):
        volume_per_class = OrderedDict()
        region_volume_per_class = OrderedDict()
        for c in all_classes:
            region_volume_per_class[c] = []
            volume_per_class[c] = np.sum(seg == c) * vol_per_voxel
            labelmap, numregions = label(seg == c, return_num=True)
            for l in range(1, numregions + 1):
                region_volume_per_class[c].append(np.sum(labelmap == l) * vol_per_voxel)
        return volume_per_class, region_volume_per_class

    def _get_unique_labels(self, patient_identifier):
        seg = np.load(join(self.folder_with_cropped_data, patient_identifier) + ".npz")['data'][-1]
        unique_classes = np.unique(seg)
        return unique_classes

    def _load_seg_analyze_classes(self, patient_identifier, all_classes):
        """
        1) what class is in this training case?
        2) what is the size distribution for each class?
        3) what is the region size of each class?
        4) check if all in one region
        :return:
        """
        seg = np.load(join(self.folder_with_cropped_data, patient_identifier) + ".npz")['data'][-1]
        pkl = load_pickle(join(self.folder_with_cropped_data, patient_identifier) + ".pkl")
        vol_per_voxel = np.prod(pkl['itk_spacing'])

        # ad 1)
        unique_classes = np.unique(seg)

        # 4) check if all in one region
        regions = list()
        regions.append(list(all_classes))
        for c in all_classes:
            regions.append((c, ))

        all_in_one_region = self._check_if_all_in_one_region(seg, regions)

        # 2 & 3) region sizes
        volume_per_class, region_sizes = self._collect_class_and_region_sizes(seg, all_classes, vol_per_voxel)

        return unique_classes, all_in_one_region, volume_per_class, region_sizes

    def get_classes(self):
        datasetjson = load_json(join(self.folder_with_cropped_data, "dataset.json"))
        return datasetjson['labels']

    def analyse_segmentations(self):
        class_dct = self.get_classes()

        if self.overwrite or not isfile(self.props_per_case_file):
            p = Pool(self.num_processes)
            res = p.map(self._get_unique_labels, self.patient_identifiers)
            p.close()
            p.join()

            props_per_patient = OrderedDict()
            for p, unique_classes in \
                            zip(self.patient_identifiers, res):
                props = dict()
                props['has_classes'] = unique_classes
                props_per_patient[p] = props

            save_pickle(props_per_patient, self.props_per_case_file)
        else:
            props_per_patient = load_pickle(self.props_per_case_file)
        return class_dct, props_per_patient

    def get_sizes_and_spacings_after_cropping(self):
        sizes = []
        spacings = []
        # for c in case_identifiers:
        for c in self.patient_identifiers:
            properties = self.load_properties_of_cropped(c)
            sizes.append(properties["size_after_cropping"])
            spacings.append(properties["original_spacing"])
            #spacings the distance between each pixel

        return sizes, spacings

    def get_modalities(self):
        datasetjson = load_json(join(self.folder_with_cropped_data, "dataset.json"))
        modalities = datasetjson["modality"]
        modalities = {int(k): modalities[k] for k in modalities.keys()}
        return modalities

    def get_size_reduction_by_cropping(self):
        size_reduction = OrderedDict()
        for p in self.patient_identifiers:
            props = self.load_properties_of_cropped(p)
            shape_before_crop = props["original_size_of_raw_data"]
            shape_after_crop = props['size_after_cropping']
            size_red = np.prod(shape_after_crop) / np.prod(shape_before_crop)
            size_reduction[p] = size_red
        return size_reduction

    def _get_voxels_in_foreground(self, patient_identifier, modality_id):
        all_data = np.load(join(self.folder_with_cropped_data, patient_identifier) + ".npz")['data']
        modality = all_data[modality_id]
        mask = all_data[-1] > 0
        voxels = list(modality[mask][::10]) # no need to take every voxel
        return voxels

    @staticmethod
    def _compute_stats(voxels):
        if len(voxels) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        median = np.median(voxels)
        mean = np.mean(voxels)
        sd = np.std(voxels)
        mn = np.min(voxels)
        mx = np.max(voxels)
        percentile_99_5 = np.percentile(voxels, 100)
        percentile_00_5 = np.percentile(voxels, 0)
        return median, mean, sd, mn, mx, percentile_99_5, percentile_00_5

    def collect_intensity_properties(self, num_modalities):
        if self.overwrite or not isfile(self.intensityproperties_file):
            p = Pool(self.num_processes)

            results = OrderedDict()
            for mod_id in range(num_modalities):
                results[mod_id] = OrderedDict()
                v = p.starmap(self._get_voxels_in_foreground, zip(self.patient_identifiers,
                                                              [mod_id] * len(self.patient_identifiers)))

                w = []
                for iv in v:
                    w += iv

                median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = self._compute_stats(w)

                local_props = p.map(self._compute_stats, v)
                props_per_case = OrderedDict()
                for i, pat in enumerate(self.patient_identifiers):
                    props_per_case[pat] = OrderedDict()
                    props_per_case[pat]['median'] = local_props[i][0]
                    props_per_case[pat]['mean'] = local_props[i][1]
                    props_per_case[pat]['sd'] = local_props[i][2]
                    props_per_case[pat]['mn'] = local_props[i][3]
                    props_per_case[pat]['mx'] = local_props[i][4]
                    props_per_case[pat]['percentile_99_5'] = local_props[i][5]
                    props_per_case[pat]['percentile_00_5'] = local_props[i][6]

                results[mod_id]['local_props'] = props_per_case
                results[mod_id]['median'] = median
                results[mod_id]['mean'] = mean
                results[mod_id]['sd'] = sd
                results[mod_id]['mn'] = mn
                results[mod_id]['mx'] = mx
                results[mod_id]['percentile_99_5'] = percentile_99_5
                results[mod_id]['percentile_00_5'] = percentile_00_5

            p.close()
            p.join()
            save_pickle(results, self.intensityproperties_file)
        else:
            results = load_pickle(self.intensityproperties_file)
        return results

    def analyze_dataset(self, collect_intensityproperties=True):
        # get all spacings and sizes
        sizes, spacings = self.get_sizes_and_spacings_after_cropping()
        # get all classes and what classes are in what patients
        # class min size
        # region size per class
        classes = self.get_classes()
        # get classes info
        all_classes = [int(i) for i in classes.keys() if int(i) > 0]

        # modalities
        modalities = self.get_modalities()

        # collect intensity information
        if collect_intensityproperties:
            intensityproperties = self.collect_intensity_properties(len(modalities))
        else:
            intensityproperties = None

        # size reduction by cropping
        size_reductions = self.get_size_reduction_by_cropping()


        dataset_properties = dict()
        dataset_properties['all_sizes'] = sizes
        dataset_properties['all_spacings'] = spacings
        dataset_properties['all_classes'] = all_classes
        dataset_properties['modalities'] = modalities  # {idx: modality name}
        dataset_properties['intensityproperties'] = intensityproperties
        dataset_properties['size_reductions'] = size_reductions  # {patient_id: size_reduction}

        save_pickle(dataset_properties, join(self.folder_with_cropped_data, "dataset_properties.pkl"))
        return dataset_properties

def get_patient_identifiers_from_cropped_files(folder):
    return [i.split("/")[-1][:-4] for i in subfiles(folder, join=True, suffix=".npz")]


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


def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    print(data)
    return nonzero_mask

def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]



def get_case_identifier(case):
    case_identifier = case[0].split("/")[-1].split(".nii.gz")[0][:-5]
    return case_identifier

def load_case_from_list_of_files(data_files, seg_file=None):
    assert isinstance(data_files, list) or isinstance(data_files, tuple), "case must be either a list or a tuple"
    properties = OrderedDict()
    data_itk = [sitk.ReadImage(f) for f in data_files]

    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
    properties["list_of_data_files"] = data_files
    properties["seg_file"] = seg_file

    properties["itk_origin"] = data_itk[0].GetOrigin()
    properties["itk_spacing"] = data_itk[0].GetSpacing()
    properties["itk_direction"] = data_itk[0].GetDirection()

    data_npy = np.vstack([sitk.GetArrayFromImage(d)[None] for d in data_itk])
    if seg_file is not None:
        seg_itk = sitk.ReadImage(seg_file)
        seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.float32)
    else:
        seg_npy = None
    return data_npy.astype(np.float32), seg_npy, properties


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask, 0)

    cropped_data = []
    for c in range(data.shape[0]):
        cropped = crop_to_bbox(data[c], bbox)
        cropped_data.append(cropped[None])
    data = np.vstack(cropped_data)

    if seg is not None:
        cropped_seg = []
        for c in range(seg.shape[0]):
            cropped = crop_to_bbox(seg[c], bbox)
            cropped_seg.append(cropped[None])
        seg = np.vstack(cropped_seg)

    nonzero_mask = crop_to_bbox(nonzero_mask, bbox)[None]
    if seg is not None:
        seg[(seg == 0) & (nonzero_mask == 0)] = nonzero_label
    else:
        nonzero_mask = nonzero_mask.astype(int)
        nonzero_mask[nonzero_mask == 0] = nonzero_label
        nonzero_mask[nonzero_mask > 0] = 0
        seg = nonzero_mask
    return data, seg, bbox

class ImageCropper(object):
    def __init__(self, num_threads, output_folder=None):
        """
        This one finds a mask of nonzero elements (must be nonzero in all modalities) and crops the image to that mask.
        In the case of BRaTS and ISLES data this results in a significant reduction in image size
        :param num_threads:
        :param output_folder: whete to store the cropped data
        :param list_of_files:
        """
        self.output_folder = output_folder
        self.num_threads = num_threads

        if self.output_folder is not None:
            maybe_mkdir_p(self.output_folder)

    @staticmethod
    def crop(data, properties, seg=None):
        shape_before = data.shape
        data, seg, bbox = crop_to_nonzero(data, seg, nonzero_label=-1)
        shape_after = data.shape
        print("before crop:", shape_before, "after crop:", shape_after, "spacing:",
              np.array(properties["original_spacing"]), "\n")
        properties["crop_bbox"] = bbox
        properties['classes'] = np.unique(seg)
        seg[seg < -1] = 0
        properties["size_after_cropping"] = data[0].shape
        return data, seg, properties

    @staticmethod
    def crop_from_list_of_files(data_files, seg_file=None):
        data, seg, properties = load_case_from_list_of_files(data_files, seg_file)
        return ImageCropper.crop(data, properties, seg)

    def load_crop_save(self, case, case_identifier, overwrite_existing=False):
        try:
            #print(case_identifier)
            if overwrite_existing \
                    or (not os.path.isfile(os.path.join(self.output_folder, "%s.npz" % case_identifier))
                        or not os.path.isfile(os.path.join(self.output_folder, "%s.pkl" % case_identifier))):

                data, seg, properties = self.crop_from_list_of_files(case[:-1], case[-1])

                all_data = np.vstack((data, seg))
                np.savez_compressed(os.path.join(self.output_folder, "%s.npz" % case_identifier), data=all_data)
                with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'wb') as f:
                    pickle.dump(properties, f)
        except Exception as e:
            print("Exception in", case_identifier, ":")
            print(e)
            raise e


    def get_list_of_cropped_files(self):
        return subfiles(self.output_folder, join=True, suffix=".npz")

    def get_patient_identifiers_from_cropped_files(self):
        return [i.split("/")[-1][:-4] for i in self.get_list_of_cropped_files()]

    def run_cropping(self, list_of_files, overwrite_existing=False, output_folder=None):
        """
        also copied ground truth nifti segmentation into the preprocessed folder so that we can use them for evaluation
        on the cluster
        :param list_of_files: list of list of files [[PATIENTID_TIMESTEP_0000.nii.gz], [PATIENTID_TIMESTEP_0000.nii.gz]]
        :param overwrite_existing:
        :param output_folder:
        :return:
        """
        if output_folder is not None:
            self.output_folder = output_folder

        output_folder_gt = os.path.join(self.output_folder, "gt_segmentations")
        maybe_mkdir_p(output_folder_gt)

        for j, case in enumerate(list_of_files):
            if case[-1] is not None:
                shutil.copy(case[-1], output_folder_gt)

        list_of_args = []
        for j, case in enumerate(list_of_files):
            case_identifier = get_case_identifier(case)
            list_of_args.append((case, case_identifier, overwrite_existing))

        p = Pool(self.num_threads)
        p.starmap(self.load_crop_save, list_of_args)
        p.close()
        p.join()

    def load_properties(self, case_identifier):
        with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return properties

    def save_properties(self, case_identifier, properties):
        with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'wb') as f:
            pickle.dump(properties, f)



def create_lists_from_splitted_dataset(base_folder_splitted):
    lists = []

    json_file = join(base_folder_splitted, "dataset.json")
    with open(json_file) as jsn:
        d = json.load(jsn)
        training_files = d['training']
    num_modalities = len(d['modality'].keys())
    for tr in training_files:
        cur_pat = []
        for mod in range(num_modalities):
            cur_pat.append(join(base_folder_splitted, "imagesTr", tr['image'].split("/")[-1][:-7] +
                                "_%04.0d.nii.gz" % mod))
        cur_pat.append(join(base_folder_splitted, "labelsTr", tr['label'].split("/")[-1]))
        lists.append(cur_pat)
    return lists, {int(i): d['modality'][str(i)] for i in d['modality'].keys()}


def crop(task_string, override=False, num_threads=default_num_threads):
    cropped_out_dir = join(nnUNet_cropped_data, task_string)
    maybe_mkdir_p(cropped_out_dir)

    if override and isdir(cropped_out_dir):
        shutil.rmtree(cropped_out_dir)
        maybe_mkdir_p(cropped_out_dir)

    splitted_4d_output_dir_task = join(nnUNet_raw_data, task_string)
    #print(splitted_4d_output_dir_task)
    lists, _ = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)
    #print(lists)
    imgcrop = ImageCropper(num_threads, cropped_out_dir)
    imgcrop.run_cropping(lists, overwrite_existing=override)
    shutil.copy(join(nnUNet_raw_data, task_string, "dataset.json"), cropped_out_dir)

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
    
    #print(network_training_output_dir)
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



def verify_all_same_orientation(folder):
    """
    This should run after cropping
    :param folder:
    :return:
    """
    nii_files = subfiles(folder, suffix=".nii.gz", join=True)
    orientations = []
    for n in nii_files:
        img = nib.load(n)
        affine = img.affine
        orientation = nib.aff2axcodes(affine)
        orientations.append(orientation)
    # now we need to check whether they are all the same
    orientations = np.array(orientations)
    unique_orientations = np.unique(orientations, axis=0)
    all_same = len(unique_orientations) == 1
    return all_same, unique_orientations


def verify_same_geometry(img_1: sitk.Image, img_2: sitk.Image):
    ori1, spacing1, direction1, size1 = img_1.GetOrigin(), img_1.GetSpacing(), img_1.GetDirection(), img_1.GetSize()
    ori2, spacing2, direction2, size2 = img_2.GetOrigin(), img_2.GetSpacing(), img_2.GetDirection(), img_2.GetSize()
    same_ori = np.all(np.isclose(ori1, ori2))
    if not same_ori:
        print("the origin does not match between the images:")
        print(ori1)
        print(ori2)

    same_spac = np.all(np.isclose(spacing1, spacing2))
    if not same_spac:
        print("the spacing does not match between the images")
        print(spacing1)
        print(spacing2)

    same_dir = np.all(np.isclose(direction1, direction2))
    if not same_dir:
        print("the direction does not match between the images")
        print(direction1)
        print(direction2)

    same_size = np.all(np.isclose(size1, size2))
    if not same_size:
        print("the size does not match between the images")
        print(size1)
        print(size2)

    if same_ori and same_spac and same_dir and same_size:
        return True
    else:
        return False


def verify_contains_only_expected_labels(itk_img: str, valid_labels: (tuple, list)):
    img_npy = sitk.GetArrayFromImage(sitk.ReadImage(itk_img))
    uniques = np.unique(img_npy)
    invalid_uniques = [i for i in uniques if i not in valid_labels]
    if len(invalid_uniques) == 0:
        r = True
    else:
        r = False
    return r, invalid_uniques

def verify_dataset_integrity(folder):
    """
    folder needs the imagesTr, imagesTs and labelsTr subfolders. There also needs to be a dataset.json
    checks if all training cases and labels are present
    checks if all test cases (if any) are present
    for each case, checks whether all modalities apre present
    for each case, checks whether the pixel grids are aligned
    checks whether the labels really only contain values they should
    :param folder:
    :return:
    """
    assert isfile(join(folder, "dataset.json")), "There needs to be a dataset.json file in folder, folder=%s" % folder
    assert isdir(join(folder, "imagesTr")), "There needs to be a imagesTr subfolder in folder, folder=%s" % folder
    assert isdir(join(folder, "labelsTr")), "There needs to be a labelsTr subfolder in folder, folder=%s" % folder
    dataset = load_json(join(folder, "dataset.json"))
    training_cases = dataset['training']
    num_modalities = len(dataset['modality'].keys())
    test_cases = dataset['test']
    expected_train_identifiers = [i['image'].split("/")[-1][:-7] for i in training_cases]
    expected_test_identifiers = [i['image'].split("/")[-1][:-7] for i in test_cases]

    ## check training set
    nii_files_in_imagesTr = subfiles((join(folder, "imagesTr")), suffix=".nii.gz", join=False)
    nii_files_in_labelsTr = subfiles((join(folder, "labelsTr")), suffix=".nii.gz", join=False)

    label_files = []
    geometries_OK = True
    has_nan = False

    # check all cases
    if len(expected_train_identifiers) != len(np.unique(expected_train_identifiers)): raise RuntimeError("found duplicate training cases in dataset.json")

    print("Verifying training set")
    for c in expected_train_identifiers:
        print("checking case", c)
        # check if all files are present
        expected_label_file = join(folder, "labelsTr", c + ".nii.gz")
        label_files.append(expected_label_file)
        expected_image_files = [join(folder, "imagesTr", c + "_%04.0d.nii.gz" % i) for i in range(num_modalities)]
        assert isfile(expected_label_file), "could not find label file for case %s. Expected file: \n%s" % (
            c, expected_label_file)
        assert all([isfile(i) for i in
                    expected_image_files]), "some image files are missing for case %s. Expected files:\n %s" % (
            c, expected_image_files)

        # verify that all modalities and the label have the same shape and geometry.
        label_itk = sitk.ReadImage(expected_label_file)

        nans_in_seg = np.any(np.isnan(sitk.GetArrayFromImage(label_itk)))
        has_nan = has_nan | nans_in_seg
        if nans_in_seg:
            print("There are NAN values in segmentation %s" % expected_label_file)

        images_itk = [sitk.ReadImage(i) for i in expected_image_files]
        for i, img in enumerate(images_itk):
            nans_in_image = np.any(np.isnan(sitk.GetArrayFromImage(img)))
            has_nan = has_nan | nans_in_image
            same_geometry = verify_same_geometry(img, label_itk)
            if not same_geometry:
                geometries_OK = False
                print("The geometry of the image %s does not match the geometry of the label file. The pixel arrays "
                      "will not be aligned and nnU-Net cannot use this data. Please make sure your image modalities "
                      "are coregistered and have the same geometry as the label" % expected_image_files[0][:-12])
            if nans_in_image:
                print("There are NAN values in image %s" % expected_image_files[i])

        # now remove checked files from the lists nii_files_in_imagesTr and nii_files_in_labelsTr
        for i in expected_image_files:
            nii_files_in_imagesTr.remove(os.path.basename(i))
        nii_files_in_labelsTr.remove(os.path.basename(expected_label_file))

    # check for stragglers
    assert len(
        nii_files_in_imagesTr) == 0, "there are training cases in imagesTr that are not listed in dataset.json: %s" % nii_files_in_imagesTr
    assert len(
        nii_files_in_labelsTr) == 0, "there are training cases in labelsTr that are not listed in dataset.json: %s" % nii_files_in_labelsTr

    # verify that only properly declared values are present in the labels
    print("Verifying label values")
    expected_labels = list(int(i) for i in dataset['labels'].keys())
    expected_labels.sort()

    # check if labels are in consecutive order
    assert expected_labels[0] == 0, 'The first label must be 0 and maps to the background'
    labels_valid_consecutive = np.ediff1d(expected_labels) == 1
    assert all(labels_valid_consecutive), f'Labels must be in consecutive order (0, 1, 2, ...). The labels {np.array(expected_labels)[1:][~labels_valid_consecutive]} do not satisfy this restriction'

    p = Pool(default_num_threads)
    results = p.starmap(verify_contains_only_expected_labels, zip(label_files, [expected_labels] * len(label_files)))
    p.close()
    p.join()

    fail = False
    print("Expected label values are", expected_labels)
    for i, r in enumerate(results):
        if not r[0]:
            print("Unexpected labels found in file %s. Found these unexpected values (they should not be there) %s" % (
                label_files[i], r[1]))
            fail = True

    if fail:
        raise AssertionError(
            "Found unexpected labels in the training dataset. Please correct that or adjust your dataset.json accordingly")
    else:
        print("Labels OK")

    # check test set, but only if there actually is a test set
    if len(expected_test_identifiers) > 0:
        print("Verifying test set")
        nii_files_in_imagesTs = subfiles((join(folder, "imagesTs")), suffix=".nii.gz", join=False)

        for c in expected_test_identifiers:
            # check if all files are present
            expected_image_files = [join(folder, "imagesTs", c + "_%04.0d.nii.gz" % i) for i in range(num_modalities)]
            assert all([isfile(i) for i in
                        expected_image_files]), "some image files are missing for case %s. Expected files:\n %s" % (
                c, expected_image_files)

            # verify that all modalities and the label have the same geometry. We use the affine for this
            if num_modalities > 1:
                images_itk = [sitk.ReadImage(i) for i in expected_image_files]
                reference_img = images_itk[0]

                for i, img in enumerate(images_itk[1:]):
                    assert verify_same_geometry(img, reference_img), "The modalities of the image %s do not seem to be " \
                                                                     "registered. Please coregister your modalities." % (
                                                                         expected_image_files[i])

            # now remove checked files from the lists nii_files_in_imagesTr and nii_files_in_labelsTr
            for i in expected_image_files:
                nii_files_in_imagesTs.remove(os.path.basename(i))
        assert len(nii_files_in_imagesTs) == 0, "there are training cases in imagesTs that are not listed in dataset.json: %s" % nii_files_in_imagesTs

    all_same, unique_orientations = verify_all_same_orientation(join(folder, "imagesTr"))
    if not all_same:
        print(
            "WARNING: Not all images in the dataset have the same axis ordering. We very strongly recommend you correct that by reorienting the data. fslreorient2std should do the trick")
    # save unique orientations to dataset.json
    if not geometries_OK:
        raise Warning("GEOMETRY MISMATCH FOUND! CHECK THE TEXT OUTPUT! This does not cause an error at this point  but you should definitely check whether your geometries are alright!")
    else:
        print("Dataset OK")

    if has_nan:
        raise RuntimeError("Some images have nan values in them. This will break the training. See text output above to see which ones")




def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_ids", nargs="+", help="List of integers belonging to the task ids you wish to run"
                                                            " experiment planning and preprocessing for. Each of these "
                                                            "ids must, have a matching folder 'TaskXXX_' in the raw "
                                                            "data folder")
    
    parser.add_argument("-pl3d", "--planner3d", type=str, default="ExperimentPlanner3D_v21",
                        help="Name of the ExperimentPlanner class for the full resolution 3D U-Net and U-Net cascade. "
                             "Default is ExperimentPlanner3D_v21. Can be 'None', in which case these U-Nets will not be "
                             "configured")
    parser.add_argument("-pl2d", "--planner2d", type=str, default="ExperimentPlanner2D_v21",
                        help="Name of the ExperimentPlanner class for the 2D U-Net. Default is ExperimentPlanner2D_v21. "
                             "Can be 'None', in which case this U-Net will not be configured")

    parser.add_argument("-no_pp", action="store_true",
                        help="Set this flag if you dont want to run the preprocessing. If this is set then this script "
                             "will only run the experiment planning and create the plans file")
    parser.add_argument("-tl", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the low resolution data for the 3D low "
                             "resolution U-Net. This can be larger than -tf. Don't overdo it or you will run out of "
                             "RAM")
    parser.add_argument("-tf", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the full resolution data of the 2D U-Net and "
                             "3D U-Net. Don't overdo it or you will run out of RAM")
    
    
    parser.add_argument("--verify_dataset_integrity", required=False, default=False, action="store_true",
                        help="set this flag to check the dataset integrity. This is useful and should be done once for "
                             "each dataset!")
    parser.add_argument("-overwrite_plans", type=str, default=None, required=False,
                        help="Use this to specify a plans file that should be used instead of whatever nnU-Net would "
                             "configure automatically. This will overwrite everything: intensity normalization, "
                             "network architecture, target spacing etc. Using this is useful for using pretrained "
                             "model weights as this will guarantee that the network architecture on the target "
                             "dataset is the same as on the source dataset and the weights can therefore be transferred.\n"
                             "Pro tip: If you want to pretrain on Hepaticvessel and apply the result to LiTS then use "
                             "the LiTS plans to run the preprocessing of the HepaticVessel task.\n"
                             "Make sure to only use plans files that were "
                             "generated with the same number of modalities as the target dataset (LiTS -> BCV or "
                             "LiTS -> Task008_HepaticVessel is OK. BraTS -> LiTS is not (BraTS has 4 input modalities, "
                             "LiTS has just one)). Also only do things that make sense. This functionality is beta with"
                             "no support given.\n"
                             "Note that this will first print the old plans (which are going to be overwritten) and "
                             "then the new ones (provided that -no_pp was NOT set).")
    parser.add_argument("-overwrite_plans_identifier", type=str, default=None, required=False,
                        help="If you set overwrite_plans you need to provide a unique identifier so that nnUNet knows "
                             "where to look for the correct plans and data. Assume your identifier is called "
                             "IDENTIFIER, the correct training command would be:\n"
                             "'nnUNet_train CONFIG TRAINER TASKID FOLD -p nnUNetPlans_pretrained_IDENTIFIER "
                             "-pretrained_weights FILENAME'")

    args = parser.parse_args()


    #040
    task_ids = args.task_ids

    dont_run_preprocessing = args.no_pp
    tl = args.tl    #8
    tf = args.tf    #8
    planner_name3d = args.planner3d #ExperimentPlanner3D_v21
    planner_name2d = args.planner2d #ExperimentPlanner2D_v21
    if planner_name3d == "None":
        planner_name3d = None
    if planner_name2d == "None":
        planner_name2d = None
    if args.overwrite_plans is not None:
        if planner_name2d is not None:
            print("Overwriting plans only works for the 3d planner. I am setting '--planner2d' to None. This will "
                  "skip 2d planning and preprocessing.")
        assert planner_name3d == 'ExperimentPlanner3D_v21_Pretrained', "When using --overwrite_plans you need to use " \
                                                                       "'-pl3d ExperimentPlanner3D_v21_Pretrained'"
    if args.overwrite_plans is not None:
        if planner_name2d is not None:
            print("Overwriting plans only works for the 3d planner. I am setting '--planner2d' to None. This will "
                  "skip 2d planning and preprocessing.")
        assert planner_name3d == 'ExperimentPlanner3D_v21_Pretrained', "When using --overwrite_plans you need to use " \
                                                                       "'-pl3d ExperimentPlanner3D_v21_Pretrained'"
    #print(tf)
    tasks = []

    for i in task_ids:
        i = int(i)
        task_name = convert_id_to_task_name(i)
        print(task_name)
        if args.verify_dataset_integrity:
            verify_dataset_integrity(join(nnUNet_raw_data, task_name))
        crop(task_name, False, tf)
        tasks.append(task_name)

    search_in = '/home/ondemand/data/Kits/unetqkv/experiment_planning'

    for t in tasks:
        #print("\n\n\n", t)
        cropped_out_dir = os.path.join(nnUNet_cropped_data, t)
    if planner_name3d is not None:
        #planner_3d = ExperimentPlanner3D_v21(cropped_out_dir, preprocessing_output_dir)
        planner_3d = recursive_find_python_class([search_in], planner_name3d, current_module="experiment_planning")
        if planner_3d is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                               "nnunet.experiment_planning" % planner_name3d)
    else:
        planner_3d = None
    if planner_name2d is not None:
        planner_2d = recursive_find_python_class([search_in], planner_name2d, current_module="nnunet.experiment_planning")
        if planner_2d is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                               "nnunet.experiment_planning" % planner_name2d)
    else:
        planner_2d = None

    
    for t in tasks:
        print("\n\n\n", t)
        cropped_out_dir = os.path.join(nnUNet_cropped_data, t)

        

        preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, t)

        
        #splitted_4d_output_dir_task = os.path.join(nnUNet_raw_data, t)
        #lists, modalities = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

        # we need to figure out if we need the intensity propoerties. We collect them only if one of the modalities is CT
        dataset_json = load_json(join(cropped_out_dir, 'dataset.json'))
        """
        {'description': 'kidney and kidney tumor segmentation', 'labels': {'0': 'background', '1': 'Kidney', '2': 'Tumor'}, 'licence': '', 'modality': {'0': 'CT'}, 'name': 'KiTS', 'numTest': 70, 'numTraining': 140, 'reference': 'KiTS data for nnunet', 'release': '0.0', 'tensorImageSize': '4D', 'test': [{'image': './imagesTs/case_00140.nii.gz', 'label': './labelsTs/case_00140.nii.gz'}, {'image': './imagesTs/case_00141.nii.gz', 'label': './labelsTs/case_00141.nii.gz'}, {'image': './imagesTs/case_00142.nii.gz', 'label': './labelsTs/case_00142.nii.gz'}, {'image': './imagesTs/case_00143.nii.gz', 'label': './labelsTs/case_00143.nii.gz'}, {'image': './imagesTs/case_00144.nii.gz', 'label': './labelsTs/case_00144.nii.gz'}, {'image': './imagesTs/case_00145.nii.gz', 'label': './labelsTs/case_00145.nii.gz'}, {'image': './imagesTs/case_00146.nii.gz', 'label': './labelsTs/case_00146.nii.gz'}, {'image': './imagesTs/case_00147.nii.gz', 'label': './labelsTs/case_00147.nii.gz'}, {'image': './imagesTs/case_00148.nii.gz', 'label': './labelsTs/case_00148.nii.gz'}, {'image': './imagesTs/case_00149.nii.gz', 'label': './labelsTs/case_00149.nii.gz'}, {'image': './imagesTs/case_00150.nii.gz', 'label': './labelsTs/case_00150.nii.gz'}, {'image': './imagesTs/case_00151.nii.gz', 'label': './labelsTs/case_00151.nii.gz'}, {'image': './imagesTs/case_00152.nii.gz', 'label': './labelsTs/case_00152.nii.gz'}, {'image': './imagesTs/case_00153.nii.gz', 'label': './labelsTs/case_00153.nii.gz'}, {'image': './imagesTs/case_00154.nii.gz', 'label': './labelsTs/case_00154.nii.gz'}, {'image': './imagesTs/case_00155.nii.gz', 'label': './labelsTs/case_00155.nii.gz'}, {'image': './imagesTs/case_00156.nii.gz', 'label': './labelsTs/case_00156.nii.gz'}, {'image': './imagesTs/case_00157.nii.gz', 'label': './labelsTs/case_00157.nii.gz'}, {'image': './imagesTs/case_00158.nii.gz', 'label': './labelsTs/case_00158.nii.gz'}, {'image': './imagesTs/case_00159.nii.gz', 'label': './labelsTs/case_00159.nii.gz'}, {'image': './imagesTs/case_00160.nii.gz', 'label': './labelsTs/case_00160.nii.gz'}, {'image': './imagesTs/case_00161.nii.gz', 'label': './labelsTs/case_00161.nii.gz'}, {'image': './imagesTs/case_00162.nii.gz', 'label': './labelsTs/case_00162.nii.gz'}, {'image': './imagesTs/case_00163.nii.gz', 'label': './labelsTs/case_00163.nii.gz'}, {'image': './imagesTs/case_00164.nii.gz', 'label': './labelsTs/case_00164.nii.gz'}, {'image': './imagesTs/case_00165.nii.gz', 'label': './labelsTs/case_00165.nii.gz'}, {'image': './imagesTs/case_00166.nii.gz', 'label': './labelsTs/case_00166.nii.gz'}, {'image': './imagesTs/case_00167.nii.gz', 'label': './labelsTs/case_00167.nii.gz'}, {'image': './imagesTs/case_00168.nii.gz', 'label': './labelsTs/case_00168.nii.gz'}, {'image': './imagesTs/case_00169.nii.gz', 'label': './labelsTs/case_00169.nii.gz'}, {'image': './imagesTs/case_00170.nii.gz', 'label': './labelsTs/case_00170.nii.gz'}, {'image': './imagesTs/case_00171.nii.gz', 'label': './labelsTs/case_00171.nii.gz'}, {'image': './imagesTs/case_00172.nii.gz', 'label': './labelsTs/case_00172.nii.gz'}, {'image': './imagesTs/case_00173.nii.gz', 'label': './labelsTs/case_00173.nii.gz'}, {'image': './imagesTs/case_00174.nii.gz', 'label': './labelsTs/case_00174.nii.gz'}, {'image': './imagesTs/case_00175.nii.gz', 'label': './labelsTs/case_00175.nii.gz'}, {'image': './imagesTs/case_00176.nii.gz', 'label': './labelsTs/case_00176.nii.gz'}, {'image': './imagesTs/case_00177.nii.gz', 'label': './labelsTs/case_00177.nii.gz'}, {'image': './imagesTs/case_00178.nii.gz', 'label': './labelsTs/case_00178.nii.gz'}, {'image': './imagesTs/case_00179.nii.gz', 'label': './labelsTs/case_00179.nii.gz'}, {'image': './imagesTs/case_00180.nii.gz', 'label': './labelsTs/case_00180.nii.gz'}, {'image': './imagesTs/case_00181.nii.gz', 'label': './labelsTs/case_00181.nii.gz'}, {'image': './imagesTs/case_00182.nii.gz', 'label': './labelsTs/case_00182.nii.gz'}, {'image': './imagesTs/case_00183.nii.gz', 'label': './labelsTs/case_00183.nii.gz'}, {'image': './imagesTs/case_00184.nii.gz', 'label': './labelsTs/case_00184.nii.gz'}, {'image': './imagesTs/case_00185.nii.gz', 'label': './labelsTs/case_00185.nii.gz'}, {'image': './imagesTs/case_00186.nii.gz', 'label': './labelsTs/case_00186.nii.gz'}, {'image': './imagesTs/case_00187.nii.gz', 'label': './labelsTs/case_00187.nii.gz'}, {'image': './imagesTs/case_00188.nii.gz', 'label': './labelsTs/case_00188.nii.gz'}, {'image': './imagesTs/case_00189.nii.gz', 'label': './labelsTs/case_00189.nii.gz'}, {'image': './imagesTs/case_00190.nii.gz', 'label': './labelsTs/case_00190.nii.gz'}, {'image': './imagesTs/case_00191.nii.gz', 'label': './labelsTs/case_00191.nii.gz'}, {'image': './imagesTs/case_00192.nii.gz', 'label': './labelsTs/case_00192.nii.gz'}, {'image': './imagesTs/case_00193.nii.gz', 'label': './labelsTs/case_00193.nii.gz'}, {'image': './imagesTs/case_00194.nii.gz', 'label': './labelsTs/case_00194.nii.gz'}, {'image': './imagesTs/case_00195.nii.gz', 'label': './labelsTs/case_00195.nii.gz'}, {'image': './imagesTs/case_00196.nii.gz', 'label': './labelsTs/case_00196.nii.gz'}, {'image': './imagesTs/case_00197.nii.gz', 'label': './labelsTs/case_00197.nii.gz'}, {'image': './imagesTs/case_00198.nii.gz', 'label': './labelsTs/case_00198.nii.gz'}, {'image': './imagesTs/case_00199.nii.gz', 'label': './labelsTs/case_00199.nii.gz'}, {'image': './imagesTs/case_00200.nii.gz', 'label': './labelsTs/case_00200.nii.gz'}, {'image': './imagesTs/case_00201.nii.gz', 'label': './labelsTs/case_00201.nii.gz'}, {'image': './imagesTs/case_00202.nii.gz', 'label': './labelsTs/case_00202.nii.gz'}, {'image': './imagesTs/case_00203.nii.gz', 'label': './labelsTs/case_00203.nii.gz'}, {'image': './imagesTs/case_00204.nii.gz', 'label': './labelsTs/case_00204.nii.gz'}, {'image': './imagesTs/case_00205.nii.gz', 'label': './labelsTs/case_00205.nii.gz'}, {'image': './imagesTs/case_00206.nii.gz', 'label': './labelsTs/case_00206.nii.gz'}, {'image': './imagesTs/case_00207.nii.gz', 'label': './labelsTs/case_00207.nii.gz'}, {'image': './imagesTs/case_00208.nii.gz', 'label': './labelsTs/case_00208.nii.gz'}, {'image': './imagesTs/case_00209.nii.gz', 'label': './labelsTs/case_00209.nii.gz'}], 'training': [{'image': './imagesTr/case_00000.nii.gz', 'label': './labelsTr/case_00000.nii.gz'}, {'image': './imagesTr/case_00001.nii.gz', 'label': './labelsTr/case_00001.nii.gz'}, {'image': './imagesTr/case_00002.nii.gz', 'label': './labelsTr/case_00002.nii.gz'}, {'image': './imagesTr/case_00003.nii.gz', 'label': './labelsTr/case_00003.nii.gz'}, {'image': './imagesTr/case_00004.nii.gz', 'label': './labelsTr/case_00004.nii.gz'}, {'image': './imagesTr/case_00005.nii.gz', 'label': './labelsTr/case_00005.nii.gz'}, {'image': './imagesTr/case_00006.nii.gz', 'label': './labelsTr/case_00006.nii.gz'}, {'image': './imagesTr/case_00007.nii.gz', 'label': './labelsTr/case_00007.nii.gz'}, {'image': './imagesTr/case_00008.nii.gz', 'label': './labelsTr/case_00008.nii.gz'}, {'image': './imagesTr/case_00009.nii.gz', 'label': './labelsTr/case_00009.nii.gz'}, {'image': './imagesTr/case_00010.nii.gz', 'label': './labelsTr/case_00010.nii.gz'}, {'image': './imagesTr/case_00011.nii.gz', 'label': './labelsTr/case_00011.nii.gz'}, {'image': './imagesTr/case_00012.nii.gz', 'label': './labelsTr/case_00012.nii.gz'}, {'image': './imagesTr/case_00013.nii.gz', 'label': './labelsTr/case_00013.nii.gz'}, {'image': './imagesTr/case_00014.nii.gz', 'label': './labelsTr/case_00014.nii.gz'}, {'image': './imagesTr/case_00015.nii.gz', 'label': './labelsTr/case_00015.nii.gz'}, {'image': './imagesTr/case_00016.nii.gz', 'label': './labelsTr/case_00016.nii.gz'}, {'image': './imagesTr/case_00017.nii.gz', 'label': './labelsTr/case_00017.nii.gz'}, {'image': './imagesTr/case_00018.nii.gz', 'label': './labelsTr/case_00018.nii.gz'}, {'image': './imagesTr/case_00019.nii.gz', 'label': './labelsTr/case_00019.nii.gz'}, {'image': './imagesTr/case_00020.nii.gz', 'label': './labelsTr/case_00020.nii.gz'}, {'image': './imagesTr/case_00021.nii.gz', 'label': './labelsTr/case_00021.nii.gz'}, {'image': './imagesTr/case_00022.nii.gz', 'label': './labelsTr/case_00022.nii.gz'}, {'image': './imagesTr/case_00023.nii.gz', 'label': './labelsTr/case_00023.nii.gz'}, {'image': './imagesTr/case_00024.nii.gz', 'label': './labelsTr/case_00024.nii.gz'}, {'image': './imagesTr/case_00025.nii.gz', 'label': './labelsTr/case_00025.nii.gz'}, {'image': './imagesTr/case_00026.nii.gz', 'label': './labelsTr/case_00026.nii.gz'}, {'image': './imagesTr/case_00027.nii.gz', 'label': './labelsTr/case_00027.nii.gz'}, {'image': './imagesTr/case_00028.nii.gz', 'label': './labelsTr/case_00028.nii.gz'}, {'image': './imagesTr/case_00029.nii.gz', 'label': './labelsTr/case_00029.nii.gz'}, {'image': './imagesTr/case_00030.nii.gz', 'label': './labelsTr/case_00030.nii.gz'}, {'image': './imagesTr/case_00031.nii.gz', 'label': './labelsTr/case_00031.nii.gz'}, {'image': './imagesTr/case_00032.nii.gz', 'label': './labelsTr/case_00032.nii.gz'}, {'image': './imagesTr/case_00033.nii.gz', 'label': './labelsTr/case_00033.nii.gz'}, {'image': './imagesTr/case_00034.nii.gz', 'label': './labelsTr/case_00034.nii.gz'}, {'image': './imagesTr/case_00035.nii.gz', 'label': './labelsTr/case_00035.nii.gz'}, {'image': './imagesTr/case_00036.nii.gz', 'label': './labelsTr/case_00036.nii.gz'}, {'image': './imagesTr/case_00037.nii.gz', 'label': './labelsTr/case_00037.nii.gz'}, {'image': './imagesTr/case_00038.nii.gz', 'label': './labelsTr/case_00038.nii.gz'}, {'image': './imagesTr/case_00039.nii.gz', 'label': './labelsTr/case_00039.nii.gz'}, {'image': './imagesTr/case_00040.nii.gz', 'label': './labelsTr/case_00040.nii.gz'}, {'image': './imagesTr/case_00041.nii.gz', 'label': './labelsTr/case_00041.nii.gz'}, {'image': './imagesTr/case_00042.nii.gz', 'label': './labelsTr/case_00042.nii.gz'}, {'image': './imagesTr/case_00043.nii.gz', 'label': './labelsTr/case_00043.nii.gz'}, {'image': './imagesTr/case_00044.nii.gz', 'label': './labelsTr/case_00044.nii.gz'}, {'image': './imagesTr/case_00045.nii.gz', 'label': './labelsTr/case_00045.nii.gz'}, {'image': './imagesTr/case_00046.nii.gz', 'label': './labelsTr/case_00046.nii.gz'}, {'image': './imagesTr/case_00047.nii.gz', 'label': './labelsTr/case_00047.nii.gz'}, {'image': './imagesTr/case_00048.nii.gz', 'label': './labelsTr/case_00048.nii.gz'}, {'image': './imagesTr/case_00049.nii.gz', 'label': './labelsTr/case_00049.nii.gz'}, {'image': './imagesTr/case_00050.nii.gz', 'label': './labelsTr/case_00050.nii.gz'}, {'image': './imagesTr/case_00051.nii.gz', 'label': './labelsTr/case_00051.nii.gz'}, {'image': './imagesTr/case_00052.nii.gz', 'label': './labelsTr/case_00052.nii.gz'}, {'image': './imagesTr/case_00053.nii.gz', 'label': './labelsTr/case_00053.nii.gz'}, {'image': './imagesTr/case_00054.nii.gz', 'label': './labelsTr/case_00054.nii.gz'}, {'image': './imagesTr/case_00055.nii.gz', 'label': './labelsTr/case_00055.nii.gz'}, {'image': './imagesTr/case_00056.nii.gz', 'label': './labelsTr/case_00056.nii.gz'}, {'image': './imagesTr/case_00057.nii.gz', 'label': './labelsTr/case_00057.nii.gz'}, {'image': './imagesTr/case_00058.nii.gz', 'label': './labelsTr/case_00058.nii.gz'}, {'image': './imagesTr/case_00059.nii.gz', 'label': './labelsTr/case_00059.nii.gz'}, {'image': './imagesTr/case_00060.nii.gz', 'label': './labelsTr/case_00060.nii.gz'}, {'image': './imagesTr/case_00061.nii.gz', 'label': './labelsTr/case_00061.nii.gz'}, {'image': './imagesTr/case_00062.nii.gz', 'label': './labelsTr/case_00062.nii.gz'}, {'image': './imagesTr/case_00063.nii.gz', 'label': './labelsTr/case_00063.nii.gz'}, {'image': './imagesTr/case_00064.nii.gz', 'label': './labelsTr/case_00064.nii.gz'}, {'image': './imagesTr/case_00065.nii.gz', 'label': './labelsTr/case_00065.nii.gz'}, {'image': './imagesTr/case_00066.nii.gz', 'label': './labelsTr/case_00066.nii.gz'}, {'image': './imagesTr/case_00067.nii.gz', 'label': './labelsTr/case_00067.nii.gz'}, {'image': './imagesTr/case_00068.nii.gz', 'label': './labelsTr/case_00068.nii.gz'}, {'image': './imagesTr/case_00069.nii.gz', 'label': './labelsTr/case_00069.nii.gz'}, {'image': './imagesTr/case_00070.nii.gz', 'label': './labelsTr/case_00070.nii.gz'}, {'image': './imagesTr/case_00071.nii.gz', 'label': './labelsTr/case_00071.nii.gz'}, {'image': './imagesTr/case_00072.nii.gz', 'label': './labelsTr/case_00072.nii.gz'}, {'image': './imagesTr/case_00073.nii.gz', 'label': './labelsTr/case_00073.nii.gz'}, {'image': './imagesTr/case_00074.nii.gz', 'label': './labelsTr/case_00074.nii.gz'}, {'image': './imagesTr/case_00075.nii.gz', 'label': './labelsTr/case_00075.nii.gz'}, {'image': './imagesTr/case_00076.nii.gz', 'label': './labelsTr/case_00076.nii.gz'}, {'image': './imagesTr/case_00077.nii.gz', 'label': './labelsTr/case_00077.nii.gz'}, {'image': './imagesTr/case_00078.nii.gz', 'label': './labelsTr/case_00078.nii.gz'}, {'image': './imagesTr/case_00079.nii.gz', 'label': './labelsTr/case_00079.nii.gz'}, {'image': './imagesTr/case_00080.nii.gz', 'label': './labelsTr/case_00080.nii.gz'}, {'image': './imagesTr/case_00081.nii.gz', 'label': './labelsTr/case_00081.nii.gz'}, {'image': './imagesTr/case_00082.nii.gz', 'label': './labelsTr/case_00082.nii.gz'}, {'image': './imagesTr/case_00083.nii.gz', 'label': './labelsTr/case_00083.nii.gz'}, {'image': './imagesTr/case_00084.nii.gz', 'label': './labelsTr/case_00084.nii.gz'}, {'image': './imagesTr/case_00085.nii.gz', 'label': './labelsTr/case_00085.nii.gz'}, {'image': './imagesTr/case_00086.nii.gz', 'label': './labelsTr/case_00086.nii.gz'}, {'image': './imagesTr/case_00087.nii.gz', 'label': './labelsTr/case_00087.nii.gz'}, {'image': './imagesTr/case_00088.nii.gz', 'label': './labelsTr/case_00088.nii.gz'}, {'image': './imagesTr/case_00089.nii.gz', 'label': './labelsTr/case_00089.nii.gz'}, {'image': './imagesTr/case_00090.nii.gz', 'label': './labelsTr/case_00090.nii.gz'}, {'image': './imagesTr/case_00091.nii.gz', 'label': './labelsTr/case_00091.nii.gz'}, {'image': './imagesTr/case_00092.nii.gz', 'label': './labelsTr/case_00092.nii.gz'}, {'image': './imagesTr/case_00093.nii.gz', 'label': './labelsTr/case_00093.nii.gz'}, {'image': './imagesTr/case_00094.nii.gz', 'label': './labelsTr/case_00094.nii.gz'}, {'image': './imagesTr/case_00095.nii.gz', 'label': './labelsTr/case_00095.nii.gz'}, {'image': './imagesTr/case_00096.nii.gz', 'label': './labelsTr/case_00096.nii.gz'}, {'image': './imagesTr/case_00097.nii.gz', 'label': './labelsTr/case_00097.nii.gz'}, {'image': './imagesTr/case_00098.nii.gz', 'label': './labelsTr/case_00098.nii.gz'}, {'image': './imagesTr/case_00099.nii.gz', 'label': './labelsTr/case_00099.nii.gz'}, {'image': './imagesTr/case_00100.nii.gz', 'label': './labelsTr/case_00100.nii.gz'}, {'image': './imagesTr/case_00101.nii.gz', 'label': './labelsTr/case_00101.nii.gz'}, {'image': './imagesTr/case_00102.nii.gz', 'label': './labelsTr/case_00102.nii.gz'}, {'image': './imagesTr/case_00103.nii.gz', 'label': './labelsTr/case_00103.nii.gz'}, {'image': './imagesTr/case_00104.nii.gz', 'label': './labelsTr/case_00104.nii.gz'}, {'image': './imagesTr/case_00105.nii.gz', 'label': './labelsTr/case_00105.nii.gz'}, {'image': './imagesTr/case_00106.nii.gz', 'label': './labelsTr/case_00106.nii.gz'}, {'image': './imagesTr/case_00107.nii.gz', 'label': './labelsTr/case_00107.nii.gz'}, {'image': './imagesTr/case_00108.nii.gz', 'label': './labelsTr/case_00108.nii.gz'}, {'image': './imagesTr/case_00109.nii.gz', 'label': './labelsTr/case_00109.nii.gz'}, {'image': './imagesTr/case_00110.nii.gz', 'label': './labelsTr/case_00110.nii.gz'}, {'image': './imagesTr/case_00111.nii.gz', 'label': './labelsTr/case_00111.nii.gz'}, {'image': './imagesTr/case_00112.nii.gz', 'label': './labelsTr/case_00112.nii.gz'}, {'image': './imagesTr/case_00113.nii.gz', 'label': './labelsTr/case_00113.nii.gz'}, {'image': './imagesTr/case_00114.nii.gz', 'label': './labelsTr/case_00114.nii.gz'}, {'image': './imagesTr/case_00115.nii.gz', 'label': './labelsTr/case_00115.nii.gz'}, {'image': './imagesTr/case_00116.nii.gz', 'label': './labelsTr/case_00116.nii.gz'}, {'image': './imagesTr/case_00117.nii.gz', 'label': './labelsTr/case_00117.nii.gz'}, {'image': './imagesTr/case_00118.nii.gz', 'label': './labelsTr/case_00118.nii.gz'}, {'image': './imagesTr/case_00119.nii.gz', 'label': './labelsTr/case_00119.nii.gz'}, {'image': './imagesTr/case_00120.nii.gz', 'label': './labelsTr/case_00120.nii.gz'}, {'image': './imagesTr/case_00121.nii.gz', 'label': './labelsTr/case_00121.nii.gz'}, {'image': './imagesTr/case_00122.nii.gz', 'label': './labelsTr/case_00122.nii.gz'}, {'image': './imagesTr/case_00123.nii.gz', 'label': './labelsTr/case_00123.nii.gz'}, {'image': './imagesTr/case_00124.nii.gz', 'label': './labelsTr/case_00124.nii.gz'}, {'image': './imagesTr/case_00125.nii.gz', 'label': './labelsTr/case_00125.nii.gz'}, {'image': './imagesTr/case_00126.nii.gz', 'label': './labelsTr/case_00126.nii.gz'}, {'image': './imagesTr/case_00127.nii.gz', 'label': './labelsTr/case_00127.nii.gz'}, {'image': './imagesTr/case_00128.nii.gz', 'label': './labelsTr/case_00128.nii.gz'}, {'image': './imagesTr/case_00129.nii.gz', 'label': './labelsTr/case_00129.nii.gz'}, {'image': './imagesTr/case_00130.nii.gz', 'label': './labelsTr/case_00130.nii.gz'}, {'image': './imagesTr/case_00131.nii.gz', 'label': './labelsTr/case_00131.nii.gz'}, {'image': './imagesTr/case_00132.nii.gz', 'label': './labelsTr/case_00132.nii.gz'}, {'image': './imagesTr/case_00133.nii.gz', 'label': './labelsTr/case_00133.nii.gz'}, {'image': './imagesTr/case_00134.nii.gz', 'label': './labelsTr/case_00134.nii.gz'}, {'image': './imagesTr/case_00135.nii.gz', 'label': './labelsTr/case_00135.nii.gz'}, {'image': './imagesTr/case_00136.nii.gz', 'label': './labelsTr/case_00136.nii.gz'}, {'image': './imagesTr/case_00137.nii.gz', 'label': './labelsTr/case_00137.nii.gz'}, {'image': './imagesTr/case_00138.nii.gz', 'label': './labelsTr/case_00138.nii.gz'}, {'image': './imagesTr/case_00139.nii.gz', 'label': './labelsTr/case_00139.nii.gz'}]}
        """
        modalities = list(dataset_json["modality"].values())
        #CT
        collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False
        #CT
        dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False, num_processes=tf)  # this class creates the fingerprint
        #dataset_analyzer
        _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)  # this will write output files that will be used by the ExperimentPlanner
        

        maybe_mkdir_p(preprocessing_output_dir_this_task)
        shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)

        shutil.copy(join(nnUNet_raw_data, t, "dataset.json"), preprocessing_output_dir_this_task)

        
        threads = (tl, tf)
        print("ddddddddd")
        print("number of threads: ", threads, "\n")
        #sys.exit(planner_3d)
        if planner_3d is not None:
            if args.overwrite_plans is not None:
                assert args.overwrite_plans_identifier is not None, "You need to specify -overwrite_plans_identifier"
                exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task, args.overwrite_plans,
                                         args.overwrite_plans_identifier)
            else:
                exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task)
            exp_planner.plan_experiment()
            if not dont_run_preprocessing:  # double negative, yooo
                exp_planner.run_preprocessing(threads)
        if planner_2d is not None:
            exp_planner = planner_2d(cropped_out_dir, preprocessing_output_dir_this_task)
            exp_planner.plan_experiment()
            if not dont_run_preprocessing:  # double negative, yooo
                exp_planner.run_preprocessing(threads)

if __name__ == "__main__":
    main()