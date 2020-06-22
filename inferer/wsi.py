
"""infer.py

Usage:
  infer.py [--gpu=<id>] [--mode=<mode>] [--model=<path>] [--batch_size=<n>] [--num_workers=<n>] [--input_dir=<path>] [--output_dir=<path>] [--tile_size=<n>] [--return_masks]
  infer.py (-h | --help)
  infer.py --version

Options:
  -h --help            Show this string.
  --version            Show version.
  --gpu=<id>           GPU list. [default: 0]
  --mode=<mode>        Inference mode. 'tile' or 'wsi'. [default: tile]
  --model=<path>       Path to saved checkpoint.
  --input_dir=<path>   Directory containing input images/WSIs.
  --output_dir=<path>  Directory where the output will be saved. [default: output/]
  --batch_size=<n>     Batch size. [default: 25]
  --num_workers=<n>    Number of workers. [default: 12]
  --tile_size=<n>      Size of tiles (assumes square shape). [default: 20000]
  --return_masks       Whether to return cropped nuclei masks
"""
import warnings
warnings.filterwarnings('ignore') 

import time
from multiprocessing import Pool, Lock
import multiprocessing as mp
mp.set_start_method('spawn', True) # ! must be at top for VScode debugging
import argparse
import glob
from importlib import import_module
import math
import os
import sys
import re

import cv2
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data
from docopt import docopt
import tqdm
import psutil
from dataloader.infer_loader import SerializeFileList, SerializeArray
from functools import reduce

from misc.utils import rm_n_mkdir, cropping_center, get_bounding_box
from postproc import hover

from . import base
import openslide

thread_lock = Lock()
####
def _init_worker_child(lock_):
    global lock
    lock = lock_
####
def _remove_inst(inst_map, remove_id_list):
    for inst_id in remove_id_list:
        inst_map[inst_map == inst_id] = 0
    return inst_map
####
def _get_patch_top_left_info(img_shape, input_size, output_size):
    in_out_diff = input_size - output_size
    nr_step = np.floor((img_shape - in_out_diff) / output_size) + 1
    last_output_coord = (in_out_diff // 2) + (nr_step) * output_size
    # generating subpatches index from orginal
    output_tl_y_list = np.arange(in_out_diff[0] // 2, last_output_coord[0], output_size[0], dtype=np.int32)
    output_tl_x_list = np.arange(in_out_diff[1] // 2, last_output_coord[1], output_size[1], dtype=np.int32)
    output_tl_y_list, output_tl_x_list = np.meshgrid(output_tl_y_list, output_tl_x_list)
    output_tl = np.stack([output_tl_y_list.flatten(), output_tl_x_list.flatten()], axis=-1)
    input_tl = output_tl - in_out_diff // 2
    return input_tl, output_tl
#### all must be np.array
def _get_tile_info(img_shape, tile_shape, ambiguous_size=128):
    # * get normal tiling set
    tile_grid_top_left, _ = _get_patch_top_left_info(img_shape, tile_shape, tile_shape)
    tile_grid_bot_right = []
    for idx in list(range(tile_grid_top_left.shape[0])):
        tile_tl = tile_grid_top_left[idx][:2]
        tile_br = tile_tl + tile_shape
        axis_sel = tile_br > img_shape
        tile_br[axis_sel] = img_shape[axis_sel]
        tile_grid_bot_right.append(tile_br)
    tile_grid_bot_right = np.array(tile_grid_bot_right)
    tile_grid = np.stack([tile_grid_top_left, tile_grid_bot_right], axis=1)
    tile_grid_x = np.unique(tile_grid_top_left[:,1])
    tile_grid_y = np.unique(tile_grid_top_left[:,0])
    # * get tiling set to fix vertical and horizontal boundary between tiles
    # for sanity, expand at boundary `ambiguous_size` to both side vertical and horizontal
    stack_coord = lambda x: np.stack([x[0].flatten(), x[1].flatten()], axis=-1)
    tile_boundary_x_top_left  = np.meshgrid(tile_grid_y, tile_grid_x[1:] - ambiguous_size)
    tile_boundary_x_bot_right = np.meshgrid(tile_grid_y + tile_shape[0], tile_grid_x[1:] + ambiguous_size)
    tile_boundary_x_top_left  = stack_coord(tile_boundary_x_top_left)
    tile_boundary_x_bot_right = stack_coord(tile_boundary_x_bot_right)
    tile_boundary_x = np.stack([tile_boundary_x_top_left, tile_boundary_x_bot_right], axis=1)
    #
    tile_boundary_y_top_left  = np.meshgrid(tile_grid_y[1:] - ambiguous_size, tile_grid_x)
    tile_boundary_y_bot_right = np.meshgrid(tile_grid_y[1:] + ambiguous_size, tile_grid_x+tile_shape[1])
    tile_boundary_y_top_left  = stack_coord(tile_boundary_y_top_left)
    tile_boundary_y_bot_right = stack_coord(tile_boundary_y_bot_right)
    tile_boundary_y = np.stack([tile_boundary_y_top_left, tile_boundary_y_bot_right], axis=1)
    tile_boundary = np.concatenate([tile_boundary_x, tile_boundary_y], axis=0)
    # * get tiling set to fix the intersection of 4 tiles
    tile_cross_top_left  = np.meshgrid(tile_grid_y[1:] -  2 * ambiguous_size, tile_grid_x[1:] - 2 * ambiguous_size)
    tile_cross_bot_right = np.meshgrid(tile_grid_y[1:] +  2 * ambiguous_size, tile_grid_x[1:] + 2 * ambiguous_size)
    tile_cross_top_left  = stack_coord(tile_cross_top_left)
    tile_cross_bot_right = stack_coord(tile_cross_bot_right)
    tile_cross = np.stack([tile_cross_top_left, tile_cross_bot_right], axis=1)
    return tile_grid, tile_boundary, tile_cross
### 
def _get_chunk_patch_info(img_shape, chunk_input_shape, patch_input_shape, patch_output_shape):
    round_to_multiple = lambda x, y: np.floor(x / y) * y
    patch_diff_shape = patch_input_shape - patch_output_shape

    chunk_output_shape = chunk_input_shape - patch_diff_shape
    chunk_output_shape = round_to_multiple(chunk_output_shape, patch_output_shape).astype(np.int64)
    chunk_input_shape  = (chunk_output_shape + patch_diff_shape).astype(np.int64)

    patch_input_tl_list, _ = _get_patch_top_left_info(img_shape, patch_input_shape, patch_output_shape)
    patch_input_br_list = patch_input_tl_list + patch_input_shape
    patch_output_tl_list = patch_input_tl_list + patch_diff_shape
    patch_output_br_list = patch_output_tl_list + patch_output_shape 
    patch_info_list = np.stack(
                        [np.stack([patch_input_tl_list, patch_input_br_list], axis=1),
                         np.stack([patch_output_tl_list, patch_output_br_list], axis=1)], axis=1)

    chunk_input_tl_list, _ = _get_patch_top_left_info(img_shape, chunk_input_shape, chunk_output_shape)
    chunk_input_br_list = chunk_input_tl_list + chunk_input_shape
    # * correct the coord so it stay within source image
    y_sel = np.nonzero(chunk_input_br_list[:,0] > img_shape[0])[0]
    x_sel = np.nonzero(chunk_input_br_list[:,1] > img_shape[1])[0]
    chunk_input_br_list[y_sel, 0] = (img_shape[0] - patch_diff_shape[0]) - chunk_input_tl_list[y_sel,0] 
    chunk_input_br_list[x_sel, 1] = (img_shape[1] - patch_diff_shape[1]) - chunk_input_tl_list[x_sel,1] 
    chunk_input_br_list[y_sel, 0] = round_to_multiple(chunk_input_br_list[y_sel, 0], patch_output_shape[0]) 
    chunk_input_br_list[x_sel, 1] = round_to_multiple(chunk_input_br_list[x_sel, 1], patch_output_shape[1]) 
    chunk_input_br_list[y_sel, 0] += chunk_input_tl_list[y_sel, 0] + patch_diff_shape[0]
    chunk_input_br_list[x_sel, 1] += chunk_input_tl_list[x_sel, 1] + patch_diff_shape[1]
    chunk_output_tl_list = chunk_input_tl_list + patch_diff_shape // 2
    chunk_output_br_list = chunk_input_br_list - patch_diff_shape // 2 # may off pixels
    chunk_info_list = np.stack(
                        [np.stack([chunk_input_tl_list , chunk_input_br_list], axis=1),
                         np.stack([chunk_output_tl_list, chunk_output_br_list], axis=1)], axis=1)

    return chunk_info_list, patch_info_list
####
# def _post_proc_para_wrapper(mmap_ptr, tile_info, func, func_kwargs):
def _post_proc_para_wrapper(pred_map_mmap_path, tile_info, func, func_kwargs):
    # print('erex')
    idx, tile_tl, tile_br = tile_info
    wsi_pred_map_ptr = np.load(pred_map_mmap_path, mmap_mode='r')
    tile_pred_map = wsi_pred_map_ptr[tile_tl[0] : tile_br[0],
                                     tile_tl[1] : tile_br[1]]
    tile_pred_map = np.array(tile_pred_map) # from mmap to ram
    # print('erey')
    return func(tile_pred_map, **func_kwargs), tile_info
####
class Inferer(base.Inferer):

    def __run_model(self, array_data, patch_top_left_list):
        dataset = SerializeArray(array_data, patch_top_left_list, self.patch_input_shape)
        # TODO: memory sharing problem
        dataloader = data.DataLoader(dataset,
                            num_workers=self.nr_worker,
                            batch_size=self.batch_size,
                            drop_last=False)

        pbar = tqdm.tqdm(desc='Process Patches', leave=True,
                    total=int(len(dataloader)), 
                    ncols=80, ascii=True, position=0)

        accumulated_patch_output = []
        for batch_idx, batch_data in enumerate(dataloader):
            sample_data_list, sample_info_list = batch_data
            sample_output_list = self.run_step(sample_data_list)
            sample_info_list = sample_info_list.numpy()
            curr_batch_size = sample_output_list.shape[0]
            sample_output_list = np.split(sample_output_list, curr_batch_size, axis=0) 
            sample_info_list = np.split(sample_info_list, curr_batch_size, axis=0)
            sample_output_list = list(zip(sample_info_list, sample_output_list))
            accumulated_patch_output.extend(sample_output_list)
            pbar.update()
        pbar.close()
        return accumulated_patch_output

    def __select_valid_patches(self, patch_info_list):
        down_sample_ratio = self.wsi_mask.shape[0] / self.wsi_proc_shape[0]
        for _ in range(len(patch_info_list)):
            patch_info = patch_info_list.pop(0)
            patch_info = np.squeeze(patch_info)
            # get the box at corresponding mag of the mask
            output_bbox = patch_info[1] * down_sample_ratio
            output_bbox = np.rint(output_bbox).astype(np.int64)
            # coord of the output of the patch (i.e center regions)
            output_roi = self.wsi_mask[output_bbox[0][0]:output_bbox[1][0],
                                       output_bbox[0][1]:output_bbox[1][1]]
            if np.sum(output_roi) > 0:
                patch_info_list.append(patch_info)
        return patch_info_list

    def __get_raw_prediction(self, chunk_info_list, patch_info_list):

        masking = lambda x, a, b: (a <= x) & (x <= b)
        for chunk_info in chunk_info_list:
            # select patch basing on top left coordinate of input
            start_coord = chunk_info[0,0]
            end_coord = chunk_info[0,1] - self.patch_input_shape
            selection = masking(patch_info_list[:,0,0,0], start_coord[0], end_coord[0]) \
                      & masking(patch_info_list[:,0,0,1], start_coord[1], end_coord[1])
            chunk_patch_info_list = np.array(patch_info_list[selection]) # * do we need copy ?
            chunk_patch_info_list = np.split(chunk_patch_info_list, chunk_patch_info_list.shape[0], axis=0)

            # further select only the patches within the provided mask
            chunk_patch_info_list = self.__select_valid_patches(chunk_patch_info_list)

            # there no valid patches, so flush 0 and skip
            if len(chunk_patch_info_list) == 0:
                self.wsi_pred_map[chunk_info[1][0][0]: chunk_info[1][1][0],
                                  chunk_info[1][0][1]: chunk_info[1][1][1]] = 0
                continue
            print(chunk_info.flatten(), 'Pass')
            # shift the coordinare from wrt slide to wrt chunk
            chunk_patch_info_list = np.array(chunk_patch_info_list)
            chunk_patch_info_list -= chunk_info[:,0]
            chunk_data = self.wsi_handler.read_region(chunk_info[0][0][::-1], self.wsi_proc_mag, 
                                                    (chunk_info[0][1] - chunk_info[0][0])[::-1])
            chunk_data = np.array(chunk_data)[...,:3]

            patch_output_list = self.__run_model(chunk_data, chunk_patch_info_list[:,0,0])

            chunk_pred_map = self.wsi_pred_map[chunk_info[1][0][0]: chunk_info[1][1][0],
                                               chunk_info[1][0][1]: chunk_info[1][1][1]] 
            for pinfo in patch_output_list:
                pcoord, pdata = pinfo
                pdata = np.squeeze(pdata)
                pcoord = np.squeeze(pcoord)[:2]
                chunk_pred_map[pcoord[0] : pcoord[0] + pdata.shape[0],
                               pcoord[1] : pcoord[1] + pdata.shape[1]] = pdata        
        return

    def __dispatch_post_processing(self, tile_info_list, callback):
        # global mmap_ptr
        # mmap_ptr = self.wsi_pred_map
        if self.nr_procs > 0: 
            proc_pool = Pool(processes=self.nr_procs, 
                            initializer=_init_worker_child, 
                            initargs=(thread_lock,))

        wsi_pred_map_mmap_path = '%s/pred_map.npy' % self.wsi_cache_path
        for idx in list(range(tile_info_list.shape[0])):
            tile_tl = tile_info_list[idx][0]
            tile_br = tile_info_list[idx][1]
            tile_info = (idx, tile_tl, tile_br)
            func_kwargs = {
                'nr_types' : None,
                'return_centroids' : True
            }

            # mp.Array()
            # TODO: standarize protocol
            if self.nr_procs > 0:
                proc_pool.apply_async(_post_proc_para_wrapper, callback=callback, 
                                    args=(wsi_pred_map_mmap_path, tile_info, 
                                        hover.process, func_kwargs))
            else:
                _post_proc_para_wrapper(wsi_pred_map_mmap_path, tile_info, 
                                        hover.process, func_kwargs)
                callback(results)
        if self.nr_procs > 0:
            proc_pool.close()
            proc_pool.join()
        return

    def process_single_file(self):
        self.nr_worker = 4
        self.batch_size = 32
        self.nr_procs = 4
        chunk_input_shape = [10000, 10000]
        ambiguous_size = 128
        # tile_shape = [4096, 4096]
        tile_shape = [2048, 2048]
        patch_input_shape = [270, 270]
        patch_output_shape = [80, 80]
        self.patch_input_shape = patch_input_shape

        # TODO: customize universal file handler to sync the protocol
        ambiguous_size = int(128)
        tile_shape = (np.array(tile_shape)).astype(np.int64)
        chunk_input_shape  = np.array(chunk_input_shape)
        patch_input_shape  = np.array(patch_input_shape)
        patch_output_shape = np.array(patch_output_shape)
        # wsi_handler = openslide.OpenSlide('dataset/home/sample.tif')
        self.wsi_handler = openslide.OpenSlide('dataset/home/sample.tif')

        # TODO: customize read lv
        self.wsi_proc_mag   = 0 # w.r.t source magnification
        self.wsi_proc_shape = self.wsi_handler.level_dimensions[self.wsi_proc_mag] # TODO: turn into func
        self.wsi_proc_shape = np.array(self.wsi_proc_shape[::-1]) # to Y, X

        self.wsi_mask = cv2.imread('dataset/home/sample.png')
        self.wsi_mask = cv2.cvtColor(self.wsi_mask, cv2.COLOR_BGR2GRAY)
        self.wsi_mask[self.wsi_mask > 0] = 1

        # * declare holder for output
        # create a memory-mapped .npy file with the predefined dimensions and dtype
        out_ch = 3
        self.wsi_inst_info  = {} 
        self.wsi_cache_path = 'dataset/home/'
        self.wsi_inst_map   = np.zeros(self.wsi_proc_shape, dtype=np.int32)
        # warning, the value within this is uninitialized
        self.wsi_pred_map = np.lib.format.open_memmap(
                                            '%s/pred_map.npy' % self.wsi_cache_path, mode='w+',
                                            shape=tuple(self.wsi_proc_shape) + (out_ch,), 
                                            dtype=np.float32)

        # * raw prediction
        start = time.perf_counter()
        # cinfo, pinfo = _get_chunk_patch_info(np.array([10, 10]), np.array([6, 6]), np.array([4,4]), np.array([2,2]))
        chunk_info_list, patch_info_list = _get_chunk_patch_info(
                                                self.wsi_proc_shape, chunk_input_shape, 
                                                patch_input_shape, patch_output_shape)
        self.__get_raw_prediction(chunk_info_list, patch_info_list)
        end = time.perf_counter()
        print(end - start)

        
        # TODO: deal with error banding
        ##### * post proc
        start = time.perf_counter()
        tile_coord_set = _get_tile_info(self.wsi_proc_shape, tile_shape, ambiguous_size)
        tile_grid_info, tile_boundary_info, tile_cross_info = tile_coord_set

        ####################### * Callback can only receive 1 arg
        def post_proc_normal_tile_callback(args):
            results, pos_args = args
            run_idx, tile_tl, tile_br = pos_args
            pred_inst, inst_info_dict = results

            if len(inst_info_dict) == 0:
                return # when there is nothing to do

            top_left = pos_args[1][::-1]
            with thread_lock:
                wsi_max_id = 0 
                if len(self.wsi_inst_info) > 0:
                    wsi_max_id = max(self.wsi_inst_info.keys()) 
                for inst_id, inst_info in inst_info_dict.items():
                    # now correct the coordinate wrt to wsi
                    inst_info['bbox']     += top_left
                    inst_info['contour']  += top_left
                    inst_info['centroid'] += top_left
                    self.wsi_inst_info[inst_id + wsi_max_id] = inst_info
                pred_inst[pred_inst > 0] += wsi_max_id
                self.wsi_inst_map[tile_tl[0] : tile_br[0],
                                  tile_tl[1] : tile_br[1]] = pred_inst
            print(pos_args)
            return
        ####################### * Callback can only receive 1 arg
        def post_proc_fixing_tile_callback(args):
            results, pos_args = args
            run_idx, tile_tl, tile_br = pos_args
            pred_inst, inst_info_dict = results

            if len(inst_info_dict) == 0:
                return # when there is nothing to do

            top_left = pos_args[1][::-1]
            with thread_lock:
                # for fixing the boundary, keep all nuclei split at boundary (i.e within unambigous region)
                # of the existing prediction map, and replace all nuclei within the region with newly predicted
                # ! must get before the removal happened
                wsi_max_id = 0 
                if len(self.wsi_inst_info) > 0:
                    wsi_max_id = max(self.wsi_inst_info.keys()) 

                # * exclude ambiguous out from old prediction map
                # check 1 pix of 4 edges to find nuclei split at boundary
                roi_inst = self.wsi_inst_map[tile_tl[0] : tile_br[0],
                                             tile_tl[1] : tile_br[1]]
                roi_inst = np.copy(roi_inst)
                roi_edge = np.concatenate([roi_inst[[0,-1],:].flatten(),
                                           roi_inst[:,[0,-1]].flatten()])
                roi_boundary_inst_list = np.unique(roi_edge)[1:] # exclude background
                roi_inner_inst_list = np.unique(roi_inst)[1:]  
                roi_inner_inst_list = np.setdiff1d(roi_inner_inst_list, 
                                                roi_boundary_inst_list, 
                                                assume_unique=True)
                roi_inst = _remove_inst(roi_inst, roi_inner_inst_list)
                self.wsi_inst_map[tile_tl[0] : tile_br[0],
                             tile_tl[1] : tile_br[1]] = roi_inst
                for inst_id in roi_inner_inst_list:
                    self.wsi_inst_info.pop(inst_id, None)

                # * exclude unambiguous out from new prediction map
                # check 1 pix of 4 edges to find nuclei split at boundary
                roi_edge = pred_inst[roi_inst > 0] # remove all overlap
                boundary_inst_list = np.unique(roi_edge) # no background to exclude                
                inner_inst_list = np.unique(pred_inst)[1:]  
                inner_inst_list = np.setdiff1d(inner_inst_list, 
                                            boundary_inst_list, 
                                            assume_unique=True)              
                pred_inst = _remove_inst(pred_inst, boundary_inst_list)

                # * proceed to overwrite
                for inst_id in inner_inst_list:
                    inst_info = inst_info_dict[inst_id]
                    # now correct the coordinate wrt to wsi
                    inst_info['bbox']     += top_left
                    inst_info['contour']  += top_left
                    inst_info['centroid'] += top_left
                    self.wsi_inst_info[inst_id + wsi_max_id] = inst_info
                pred_inst[pred_inst > 0] += wsi_max_id
                pred_inst = roi_inst + pred_inst
                self.wsi_inst_map[tile_tl[0] : tile_br[0],
                                  tile_tl[1] : tile_br[1]] = pred_inst
            print(pos_args)
            return        
        #######################
        # * must be in sequential ordering
        self.__dispatch_post_processing(tile_grid_info, post_proc_normal_tile_callback)
        self.__dispatch_post_processing(tile_boundary_info, post_proc_fixing_tile_callback)
        self.__dispatch_post_processing(tile_cross_info, post_proc_fixing_tile_callback)
        end = time.perf_counter()
        print(end - start)

        # import pickle
        # with open('dataset/home/holder_postproc_tile_grid.pickle', 'rb') as fp:
        #     wsi_inst_info = pickle.load(fp)

        # import matplotlib.pyplot as plt
        # cmap = plt.get_cmap('viridis')
        # roi = self.wsi_inst_map[15000:17000,15000:17000]
        # inst_id_list = np.unique(roi)[1:]
        # roi = np.copy(roi)
        # roi[roi > 0] -= np.min(roi[roi > 0])
        # roi_color = (cmap(roi)[...,:3] * 255).astype(np.uint8)
        # inst_contour_list = [self.wsi_inst_info[inst_id]['contour'] - np.array([15000, 15000])\
        #                          for inst_id in (inst_id_list)]
        # cv2.drawContours(roi_color, inst_contour_list, -1, (255, 0, 0), 4)
        # plt.subplot(1,2,1)
        # plt.imshow(roi)
        # plt.subplot(1,2,2)
        # plt.imshow(roi_color)
        # plt.savefig('dump.png', dpi=600)
        # plt.close()
        # print('here')
        
        exit()
        return

    def process_wsi_list(self):
        return
