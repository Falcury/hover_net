import argparse
import glob
import json
import math
import multiprocessing
import os
import re
import sys
from importlib import import_module
from multiprocessing import Lock, Pool

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import tqdm
import xml.etree.ElementTree as ET
import cv2

from run_utils.utils import convert_pytorch_checkpoint
from misc.utils import log_info


####
class InferManager(object):
    def __init__(self, **kwargs):
        self.run_step = None
        for variable, value in kwargs.items():
            self.__setattr__(variable, value)
        self.__load_model()
        self.nr_types = self.method["model_args"]["nr_types"]
        # create type info name and colour

        # default
        self.type_info_dict = {
            None: ["no label", [0, 0, 0]],
        }

        if self.nr_types is not None and self.type_info_path is not None:
            self.type_info_dict = json.load(open(self.type_info_path, "r"))
            self.type_info_dict = {
                int(k): (v[0], tuple(v[1])) for k, v in self.type_info_dict.items()
            }
            # availability check
            for k in range(self.nr_types):
                if k not in self.type_info_dict:
                    assert False, "Not detect type_id=%d defined in json." % k

        if self.nr_types is not None and self.type_info_path is None:
            cmap = plt.get_cmap("hot")
            colour_list = np.arange(self.nr_types, dtype=np.int32)
            colour_list = (cmap(colour_list)[..., :3] * 255).astype(np.uint8)
            # should be compatible out of the box wrt qupath
            self.type_info_dict = {
                k: (str(k), tuple(v)) for k, v in enumerate(colour_list)
            }
        return

    def __load_model(self):
        """Create the model, load the checkpoint and define
        associated run steps to process each data batch.
        
        """
        model_desc = import_module("models.hovernet.net_desc")
        model_creator = getattr(model_desc, "create_model")

        net = model_creator(**self.method["model_args"])
        saved_state_dict = torch.load(self.method["model_path"])["desc"]
        saved_state_dict = convert_pytorch_checkpoint(saved_state_dict)

        net.load_state_dict(saved_state_dict, strict=True)
        net = torch.nn.DataParallel(net)
        net = net.to("cuda")

        module_lib = import_module("models.hovernet.run_desc")
        run_step = getattr(module_lib, "infer_step")
        self.run_step = lambda input_batch: run_step(input_batch, net)

        module_lib = import_module("models.hovernet.post_proc")
        self.post_proc_func = getattr(module_lib, "process")
        return

    def __save_xml(self, path, json_dict, type_info_dict, with_smooth_contours=True):

        xml_template = "<ASAP_Annotations>\n<AnnotationGroups>"
        def rgb_to_hex(class_color):
            return '#%02x%02x%02x' % (class_color[0], class_color[1], class_color[2])
        hex_color_dict = {key: rgb_to_hex(value[1]) for key, value in type_info_dict.items()}

        for i, type_info in type_info_dict.items():
            xml_template += f'\n\t<Group Color="{hex_color_dict[i]}" Name="{type_info[0]}" PartOfGroup="None"><Attributes/></Group>'

        xml_template += "\n</AnnotationGroups>\n<Annotations>\n</Annotations></ASAP_Annotations>"

        tree = ET.ElementTree(ET.fromstring(xml_template))
        root = tree.getroot()
        annotations = root.find("Annotations")

        next_annot_index = 0

        for key, nucleus in json_dict.items():
            contour = nucleus["contour"]
            if with_smooth_contours:
                contour = cv2.approxPolyDP(np.array(contour, dtype=np.int32), 1,
                                           closed=True)  # smooth contours to reduce the number of vertices

            nucleus_type = int(nucleus["type"])
            class_name = type_info_dict[nucleus_type][0]
            class_color = hex_color_dict[nucleus_type]

            Annotation = ET.SubElement(annotations, "Annotation", Name="Annotation " + str(next_annot_index),
                                       Type="Polygon", PartOfGroup=class_name, Color=class_color)
            next_annot_index += 1

            Coordinates = ET.SubElement(Annotation, "Coordinates")
            coordinate_no = 0
            for coord in contour:
                x = coord[0][0]
                y = coord[0][1]
                Coordinate = ET.SubElement(Coordinates, "Coordinate", Order=str(coordinate_no), X=str(x), Y=str(y))
                coordinate_no += 1

        tree.write(path)
        # log_info(f"Saving... {path}")

    def __save_json(self, path, old_dict, mag=None):
        new_dict = {}
        for inst_id, inst_info in old_dict.items():
            new_inst_info = {}
            for info_name, info_value in inst_info.items():
                # convert to jsonable
                if isinstance(info_value, np.ndarray):
                    info_value = info_value.tolist()
                new_inst_info[info_name] = info_value
            new_dict[int(inst_id)] = new_inst_info

        json_dict = {"mag": mag, "nuc": new_dict}  # to sync the format protocol
        with open(path, "w") as handle:
            json.dump(json_dict, handle)
        return new_dict
