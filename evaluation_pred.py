from abc import ABC, abstractmethod
from typing import Tuple, List, Union, Optional
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json, \
    isfile
from multiprocessing import Pool
import multiprocessing
import SimpleITK as sitk
from collections.abc import Iterable
import torch
from copy import deepcopy

import sys


default_num_processes = 8

class BaseReaderWriter(ABC):
    @staticmethod
    def _check_all_same(input_list):
        # compare all entries to the first
        for i in input_list[1:]:
            if not len(i) == len(input_list[0]):
                return False
            all_same = all(i[j] == input_list[0][j] for j in range(len(i)))
            if not all_same:
                return False
        return True

    @staticmethod
    def _check_all_same_array(input_list):
        # compare all entries to the first
        for i in input_list[1:]:
            if not all([a == b for a, b in zip(i.shape, input_list[0].shape)]):
                return False
            all_same = np.all(np.isclose(i, input_list[0]))
            if not all_same:
                return False
        return True

    @abstractmethod
    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        """
        Reads a sequence of images and returns a 4d (!) np.ndarray along with a dictionary. The 4d array must have the
        modalities (or color channels, or however you would like to call them) in its first axis, followed by the
        spatial dimensions (so shape must be c,x,y,z where c is the number of modalities (can be 1)).
        Use the dictionary to store necessary meta information that is lost when converting to numpy arrays, for
        example the Spacing, Orientation and Direction of the image. This dictionary will be handed over to write_seg
        for exporting the predicted segmentations, so make sure you have everything you need in there!

        IMPORTANT: dict MUST have a 'spacing' key with a tuple/list of length 3 with the voxel spacing of the np.ndarray.
        Example: my_dict = {'spacing': (3, 0.5, 0.5), ...}. This is needed for planning and
        preprocessing. The ordering of the numbers must correspond to the axis ordering in the returned numpy array. So
        if the array has shape c,x,y,z and the spacing is (a,b,c) then a must be the spacing of x, b the spacing of y
        and c the spacing of z.

        In the case of 2D images, the returned array should have shape (c, 1, x, y) and the spacing should be
        (999, sp_x, sp_y). Make sure 999 is larger than sp_x and sp_y! Example: shape=(3, 1, 224, 224),
        spacing=(999, 1, 1)

        For images that don't have a spacing, set the spacing to 1 (2d exception with 999 for the first axis still applies!)

        :param image_fnames:
        :return:
            1) a np.ndarray of shape (c, x, y, z) where c is the number of image channels (can be 1) and x, y, z are
            the spatial dimensions (set x=1 for 2D! Example: (3, 1, 224, 224) for RGB image).
            2) a dictionary with metadata. This can be anything. BUT it HAS to inclue a {'spacing': (a, b, c)} where a
            is the spacing of x, b of y and c of z! If an image doesn't have spacing, just set this to 1. For 2D, set
            a=999 (largest spacing value! Make it larger than b and c)

        """
        pass

    @abstractmethod
    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        """
        Same requirements as BaseReaderWriter.read_image. Returned segmentations must have shape 1,x,y,z. Multiple
        segmentations are not (yet?) allowed

        If images and segmentations can be read the same way you can just `return self.read_image((image_fname,))`
        :param seg_fname:
        :return:
            1) a np.ndarray of shape (1, x, y, z) where x, y, z are
            the spatial dimensions (set x=1 for 2D! Example: (1, 1, 224, 224) for 2D segmentation).
            2) a dictionary with metadata. This can be anything. BUT it HAS to inclue a {'spacing': (a, b, c)} where a
            is the spacing of x, b of y and c of z! If an image doesn't have spacing, just set this to 1. For 2D, set
            a=999 (largest spacing value! Make it larger than b and c)
        """
        pass

    @abstractmethod
    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        """
        Export the predicted segmentation to the desired file format. The given seg array will have the same shape and
        orientation as the corresponding image data, so you don't need to do any resampling or whatever. Just save :-)

        properties is the same dictionary you created during read_images/read_seg so you can use the information here
        to restore metadata

        IMPORTANT: Segmentations are always 3D! If your input images were 2d then the segmentation will have shape
        1,x,y. You need to catch that and export accordingly (for 2d images you need to convert the 3d segmentation
        to 2d via seg = seg[0])!

        :param seg: A segmentation (np.ndarray, integer) of shape (x, y, z). For 2D segmentations this will be (1, y, z)!
        :param output_fname:
        :param properties: the dictionary that you created in read_images (the ones this segmentation is based on).
        Use this to restore metadata
        :return:
        """
        pass


class SimpleITKIO(BaseReaderWriter):
    supported_file_endings = [
        '.nii.gz',
        '.nrrd',
        '.mha'
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images = []
        spacings = []
        origins = []
        directions = []

        spacings_for_nnunet = []
        for f in image_fnames:
            itk_image = sitk.ReadImage(f)
            spacings.append(itk_image.GetSpacing())
            origins.append(itk_image.GetOrigin())
            directions.append(itk_image.GetDirection())
            npy_image = sitk.GetArrayFromImage(itk_image)
            if len(npy_image.shape) == 2:
                # 2d
                npy_image = npy_image[None, None]
                max_spacing = max(spacings[-1])
                spacings_for_nnunet.append((max_spacing * 999, *list(spacings[-1])[::-1]))
            elif len(npy_image.shape) == 3:
                # 3d, as in original nnunet
                npy_image = npy_image[None]
                spacings_for_nnunet.append(list(spacings[-1])[::-1])
            elif len(npy_image.shape) == 4:
                # 4d, multiple modalities in one file
                spacings_for_nnunet.append(list(spacings[-1])[1::-1])
                pass
            else:
                raise RuntimeError("Unexpected number of dimensions: %d in file %s" % (len(npy_image.shape), f))

            images.append(npy_image)
            spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(spacings):
            print('ERROR! Not all input images have the same spacing!')
            print('Spacings:')
            print(spacings)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(origins):
            print('WARNING! Not all input images have the same origin!')
            print('Origins:')
            print(origins)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNet_plot_dataset_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(directions):
            print('WARNING! Not all input images have the same direction!')
            print('Directions:')
            print(directions)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNet_plot_dataset_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(spacings_for_nnunet):
            print('ERROR! Not all input images have the same spacing_for_nnunet! (This should not happen and must be a '
                  'bug. Please report!')
            print('spacings_for_nnunet:')
            print(spacings_for_nnunet)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        stacked_images = np.vstack(images)
        dict = {
            'sitk_stuff': {
                # this saves the sitk geometry information. This part is NOT used by nnU-Net!
                'spacing': spacings[0],
                'origin': origins[0],
                'direction': directions[0]
            },
            # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
            # are returned x,y,z but spacing is returned z,y,x. Duh.
            'spacing': spacings_for_nnunet[0]
        }
        return stacked_images.astype(np.float32), dict

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname, ))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        assert len(seg.shape) == 3, 'segmentation must be 3d. If you are exporting a 2d segmentation, please provide it as shape 1,x,y'
        output_dimension = len(properties['sitk_stuff']['spacing'])
        assert 1 < output_dimension < 4
        if output_dimension == 2:
            seg = seg[0]

        itk_image = sitk.GetImageFromArray(seg.astype(np.uint8))
        itk_image.SetSpacing(properties['sitk_stuff']['spacing'])
        itk_image.SetOrigin(properties['sitk_stuff']['origin'])
        itk_image.SetDirection(properties['sitk_stuff']['direction'])

        sitk.WriteImage(itk_image, output_fname)

def label_or_region_to_key(label_or_region: Union[int, Tuple[int]]):
    return str(label_or_region)


def key_to_label_or_region(key: str):
    try:
        return int(key)
    except ValueError:
        key = key.replace('(', '')
        key = key.replace(')', '')
        splitted = key.split(',')
        return tuple([int(i) for i in splitted])



def compute_metrics(reference_file: str, prediction_file: str, image_reader_writer: BaseReaderWriter,
                    labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                    ignore_label: int = None) -> dict:
    # load images
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)
    seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)
    # spacing = seg_ref_dict['spacing']

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['metrics'] = {}
    for r in labels_or_regions:
        results['metrics'][r] = {}
        mask_ref = region_or_label_to_mask(seg_ref, r)
        mask_pred = region_or_label_to_mask(seg_pred, r)
        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
        if tp + fp + fn == 0:
            results['metrics'][r]['Dice'] = np.nan
            results['metrics'][r]['IoU'] = np.nan
        else:
            results['metrics'][r]['Dice'] = 2 * tp / (2 * tp + fp + fn)
            results['metrics'][r]['IoU'] = tp / (tp + fp + fn)
        results['metrics'][r]['FP'] = fp
        results['metrics'][r]['TP'] = tp
        results['metrics'][r]['FN'] = fn
        results['metrics'][r]['TN'] = tn
        results['metrics'][r]['n_pred'] = fp + tp
        results['metrics'][r]['n_ref'] = fn + tp
    return results


def labels_to_list_of_regions(labels: List[int]):
    return [(i,) for i in labels]

def recursive_fix_for_json_export(my_dict: dict):
    # json is stupid. 'cannot serialize object of type bool_/int64/float64'. Come on bro.
    keys = list(my_dict.keys())  # cannot iterate over keys() if we change keys....
    for k in keys:
        if isinstance(k, (np.int64, np.int32, np.int8, np.uint8)):
            tmp = my_dict[k]
            del my_dict[k]
            my_dict[int(k)] = tmp
            del tmp
            k = int(k)

        if isinstance(my_dict[k], dict):
            recursive_fix_for_json_export(my_dict[k])
        elif isinstance(my_dict[k], np.ndarray):
            assert len(my_dict[k].shape) == 1, 'only 1d arrays are supported'
            my_dict[k] = fix_types_iterable(my_dict[k], output_type=list)
        elif isinstance(my_dict[k], (np.bool_,)):
            my_dict[k] = bool(my_dict[k])
        elif isinstance(my_dict[k], (np.int64, np.int32, np.int8, np.uint8)):
            my_dict[k] = int(my_dict[k])
        elif isinstance(my_dict[k], (np.float32, np.float64, np.float16)):
            my_dict[k] = float(my_dict[k])
        elif isinstance(my_dict[k], list):
            my_dict[k] = fix_types_iterable(my_dict[k], output_type=type(my_dict[k]))
        elif isinstance(my_dict[k], tuple):
            my_dict[k] = fix_types_iterable(my_dict[k], output_type=tuple)
        elif isinstance(my_dict[k], torch.device):
            my_dict[k] = str(my_dict[k])
        else:
            pass  # pray it can be serialized

def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            if isinstance(r,int):
                mask[segmentation == r] = True
            else:
                for e in r:
                    mask[segmentation == e] = True
    return mask

def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn

def compute_metrics_on_folder(folder_ref: str, folder_pred: str, output_file: str,
                              image_reader_writer: BaseReaderWriter,
                              file_ending: str,
                              regions_or_labels: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                              ignore_label: int = None,
                              num_processes: int = default_num_processes,
                              chill: bool = True) -> dict:
    """
    output_file must end with .json; can be None
    """
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)
    if not chill:
        present = [isfile(join(folder_pred, i)) for i in files_ref]
        assert all(present), "Not all files in folder_pred exist in folder_ref"
    files_ref = [join(folder_ref, i) for i in files_pred]
    files_pred = [join(folder_pred, i) for i in files_pred]
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        # for i in list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred), [ignore_label] * len(files_pred))):
        #     compute_metrics(*i)
        results = pool.starmap(
            compute_metrics,
            list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred),
                     [ignore_label] * len(files_pred)))
        )

    # mean metric per class
    metric_list = list(results[0]['metrics'][regions_or_labels[0]].keys())
    means = {}
    for r in regions_or_labels:
        means[r] = {}
        for m in metric_list:
            means[r][m] = np.nanmean([i['metrics'][r][m] for i in results])

    # foreground mean
    foreground_mean = {}
    for m in metric_list:
        values = []
        for k in means.keys():
            if k == 0 or k == '0':
                continue
            values.append(means[k][m])
        foreground_mean[m] = np.mean(values)

    [recursive_fix_for_json_export(i) for i in results]
    recursive_fix_for_json_export(means)
    recursive_fix_for_json_export(foreground_mean)
    result = {'metric_per_case': results, 'mean': means, 'foreground_mean': foreground_mean}
    if output_file is not None:
        save_summary_json(result, output_file)
    return result
    # print('DONE')

def save_summary_json(results: dict, output_file: str):
    """
    stupid json does not support tuples as keys (why does it have to be so shitty) so we need to convert that shit
    ourselves
    """
    results_converted = deepcopy(results)
    # convert keys in mean metrics
    results_converted['mean'] = {label_or_region_to_key(k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results_converted["metric_per_case"])):
        results_converted["metric_per_case"][i]['metrics'] = \
            {label_or_region_to_key(k): results["metric_per_case"][i]['metrics'][k]
             for k in results["metric_per_case"][i]['metrics'].keys()}
    # sort_keys=True will make foreground_mean the first entry and thus easy to spot
    save_json(results_converted, output_file, sort_keys=True)


def load_summary_json(filename: str):
    results = load_json(filename)
    # convert keys in mean metrics
    results['mean'] = {key_to_label_or_region(k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results["metric_per_case"])):
        results["metric_per_case"][i]['metrics'] = \
            {key_to_label_or_region(k): results["metric_per_case"][i]['metrics'][k]
             for k in results["metric_per_case"][i]['metrics'].keys()}
    return results


if __name__ == '__main__':
    folder_ref = '/orange/Dataset220_KiTS2023/labelsTs'
    folder_pred = '/orange/unet_data/nnUNet_trained_models/nnunet_qkv50d25/test_KiTS'
    output_file = '/orange/unet_data/nnUNet_trained_models/nnunet_qkv50d25/test_KiTS.json'
    image_reader_writer = SimpleITKIO()
    file_ending = '.nii.gz'
    regions = labels_to_list_of_regions([(1,2,3),(2,3),2])
    ignore_label = None
    num_processes = 12
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, image_reader_writer, file_ending, regions, ignore_label,
                              num_processes)