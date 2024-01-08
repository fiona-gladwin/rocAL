# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import random
# from amd.rocal.plugin.pytorch import ROCALClassificationIterator

from amd.rocal.pipeline import Pipeline
from rocal_pybind import rocalTensor, rocalTensorList
import torch
import amd.rocal.fn as fn
import amd.rocal.types as types
import sys
import numpy as np
import cupy as cp
import cv2
import os
import ctypes
import rocal_pybind as b
from random import shuffle

data_dir = "/dockerx/MIVisionX-data/rocal_data/coco/coco_10_img/val_10images_2017/"
data_dir2 = "/dockerx/MIVisionX-data/rocal_data/coco/coco_10_img/train_10images_2017/"
class ROCALCustomIterator(object):
    """
    COCO ROCAL iterator for pyTorch.

    Parameters
    ----------
    pipelines : list of amd.rocal.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """

    def __init__(self, pipelines, tensor_layout=types.NCHW, reverse_channels=False, multiplier=None, offset=None, tensor_dtype=types.FLOAT, device="cpu", display=False):

        try:
            assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        except Exception as ex:
            print(ex)
        self.loader = pipelines
        self.tensor_format = tensor_layout
        self.multiplier = multiplier if multiplier else [1.0, 1.0, 1.0]
        self.offset = offset if offset else [0.0, 0.0, 0.0]
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.device = device
        self.device_id = self.loader._device_id
        self.batch_size = self.bs = self.loader._batch_size
        self.output_list = self.dimensions = self.torch_dtype = None
        self.display = display
        # Image id of a batch of images
        self.image_id = np.zeros(self.bs, dtype="int32")
        # Count of labels/ bboxes in a batch
        # self.bboxes_label_count = np.zeros(self.bs, dtype="int32")
        # Image sizes of a batch
        # self.img_size = np.zeros((self.bs * 2), dtype="int32")
        self.output_memory_type = self.loader._output_memory_type
        if self.loader._is_external_source_operator:        ## How to handle for multiple readers case
            self.eos = False
            self.index = 0
            self.num_batches = self.loader._external_source.n // self.batch_size if self.loader._external_source.n % self.batch_size == 0 else (
                self.loader._external_source.n // self.batch_size + 1)
        else:
            self.num_batches = None
    def next(self):
        return self.__next__()

    def __next__(self):
        if (self.loader._is_external_source_operator):
            if (self.index + 1) == self.num_batches:
                self.eos = True
            if (self.index + 1) <= self.num_batches:
                for external_source in self.loader._external_source_readers_list:
                    data_loader_source = next(external_source.source())
                    # Extract all data from the source
                    images_list = data_loader_source[0] if (external_source.mode() == types.EXTSOURCE_FNAME) else []
                    input_buffer = data_loader_source[0] if (external_source.mode() != types.EXTSOURCE_FNAME) else []
                    labels_data = data_loader_source[1] if (len(data_loader_source) > 1) else None
                    roi_height = data_loader_source[2] if (len(data_loader_source) > 2) else []
                    roi_width = data_loader_source[3] if (len(data_loader_source) > 3) else []
                    ROIxywh_list = []
                    for i in range(self.batch_size):
                        ROIxywh = b.ROIxywh()
                        ROIxywh.x =  0
                        ROIxywh.y =  0
                        ROIxywh.w = roi_width[i] if len(roi_width) > 0 else 0
                        ROIxywh.h = roi_height[i] if len(roi_height) > 0 else 0
                        ROIxywh_list.append(ROIxywh)
                    if (len(data_loader_source) == 6 and external_source.mode() == types.EXTSOURCE_RAW_UNCOMPRESSED):
                        decoded_height = data_loader_source[4]
                        decoded_width = data_loader_source[5]
                    else:
                        decoded_width, decoded_height = external_source.dims()

                    kwargs_pybind = {
                        "handle": self.loader._handle,
                        "source_input_images": images_list,
                        "labels": labels_data,
                        "input_batch_buffer": input_buffer,
                        "roi_xywh": ROIxywh_list,
                        "decoded_width": decoded_width,
                        "decoded_height": decoded_height,
                        "channels": 3,
                        "external_source_mode": external_source.mode(),
                        "rocal_tensor_layout": types.NCHW,
                        "eos": self.eos,
                        "loader_id":external_source.id()}
                    print("ARGUMENTS : ", kwargs_pybind)
                    b.externalSourceFeedInput(*(kwargs_pybind.values()))

            self.index = self.index + 1
        if self.loader.rocal_run() != 0:
            raise StopIteration
        self.output_tensor_list = self.loader.get_outputs()

        if self.output_list is None:
            self.output_list = []
            for i in range(len(self.output_tensor_list)):
                if  isinstance(self.output_tensor_list[i], rocalTensor):
                    self.dimensions = self.output_tensor_list[i].dimensions()
                    self.torch_dtype = self.output_tensor_list[i].dtype()
                    if self.device == "cpu":
                        self.output = torch.empty(
                            self.dimensions, dtype=getattr(torch, self.torch_dtype))
                    else:
                        torch_gpu_device = torch.device('cuda', self.device_id)
                        self.output = torch.empty(self.dimensions, dtype=getattr(
                            torch, self.torch_dtype), device=torch_gpu_device)
                    self.output_tensor_list[i].copy_data(ctypes.c_void_p(
                        self.output.data_ptr()), self.output_memory_type)
                    self.output_list.append(self.output)
                else:
                    self.output_list.append(self.output_tensor_list[i])
        else:
            for i in range(len(self.output_tensor_list)):
                if  isinstance(self.output_tensor_list[i], rocalTensor):
                    self.output_tensor_list[i].copy_data(ctypes.c_void_p(
                        self.output_list[i].data_ptr()), self.output_memory_type)
                else:
                    self.output_list[i] = self.output_tensor_list[i]

        # self.labels = self.loader.get_bounding_box_labels()
        # 1D bboxes array in a batch
        # self.bboxes = self.loader.get_bounding_box_cords()
        # self.loader.get_image_id(self.image_id)
        # image_id_tensor = torch.tensor(self.image_id)
        # image_size_tensor = torch.tensor(self.img_size).view(-1, self.bs, 2)

        for i in range(self.bs):
            self.image_id[i] = i
        if self.display:
            for i in range(self.bs):
                img = self.output
                draw_patches(img[i], self.image_id[i], self.device, self.tensor_dtype, self.tensor_format)
        return tuple(self.output_list)

    def reset(self):
        self.loader.rocal_reset_loaders()

    def __iter__(self):
        return self

##################### MODE 0 #########################
# Define the Data Source for all image samples - User needs to define their own source
class ExternalInputIteratorMode0(object):
    def __init__(self, batch_size):
        self.images_dir = data_dir
        self.batch_size = batch_size
        self.files = []
        import glob
        for filename in glob.glob(os.path.join(self.images_dir, '*.jpg')):
            self.files.append(filename)
        shuffle(self.files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []
        label = 1
        for _ in range(self.batch_size):
            jpeg_filename = self.files[self.i]
            batch.append(jpeg_filename)
            labels.append(label)
            label = label + 1
            self.i = (self.i + 1) % self.n
        labels = np.array(labels).astype('int32')
        return batch, labels

##################### MODE 1 #########################
# Define the Data Source for all image samples
class ExternalInputIteratorMode1(object):
    def __init__(self, batch_size):
        self.images_dir = data_dir2
        self.batch_size = batch_size
        self.files = []
        import os
        import glob
        for filename in glob.glob(os.path.join(self.images_dir, '*.jpg')):
            self.files.append(filename)
        print("FILES : ", self.files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        self.maxWidth = None
        self.maxHeight = None
        return self

    def __next__(self):
        batch = []
        labels = []
        srcsize_height = []
        label = 1
        print("-------------NEXT------------------")
        for x in range(self.batch_size):
            jpeg_filename = self.files[self.i]
            f = open(jpeg_filename, 'rb')
            numpy_buffer = np.frombuffer(f.read(), dtype=np.uint8)
            batch.append(numpy_buffer)
            srcsize_height.append(len(numpy_buffer))
            labels.append(label)
            label = label + 1
            self.i = (self.i + 1) % self.n
        labels = np.array(labels).astype('int32')
        print("DATA Batch : ", batch)
        return (batch, labels, srcsize_height)

def image_dump(img, idx, device="cpu", mode=0):
    if device == "gpu":
        img = cp.asnumpy(img)
    img = img.cpu().detach().numpy()
    img = img.transpose([1, 2, 0])  # NCHW
    img = (img).astype('uint8')
    if mode!=2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("OUTPUT_IMAGES_PYTHON/EXTERNAL_SOURCE_READER/MODE" + str(mode) + "/"+
                str(idx)+"_"+"train"+".png", img)

def main():
    if len(sys.argv) < 3:
        print('Please pass image_folder json_path cpu/gpu batch_size')
        exit(0)
    try:
        path = "OUTPUT_FOLDER/COCO_READER/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    image_path = sys.argv[1]
    annotation_path = sys.argv[2]
    image_path1 = "/dockerx/MIVisionX-data/rocal_data/coco/coco_10_img/val_10images_2017/"
    annotation_path1 = "/dockerx/MIVisionX-data/rocal_data/coco/coco_10_img/annotations/instances_val2017.json"
    rocal_cpu = True if sys.argv[3] == "cpu" else False
    device = sys.argv[3]
    batch_size = int(sys.argv[4])
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    local_rank = 0
    world_size = 1

# Mode 1
    eii_1 = ExternalInputIteratorMode1(batch_size)
    eii_0 = ExternalInputIteratorMode0(batch_size)

    # Create the pipeline
    external_source_pipeline_mode1 = Pipeline(batch_size=batch_size, num_threads=1, device_id=0, prefetch_queue_depth=4,
                                              seed=1, rocal_cpu=rocal_cpu, tensor_layout=types.NCHW)

    with external_source_pipeline_mode1:
        jpegs, _ = fn.external_source(
            source=eii_1, mode=types.EXTSOURCE_RAW_COMPRESSED, max_width=2000, max_height=2000)
        output = fn.resize(jpegs, resize_width=200, resize_height=200,
                           output_layout=types.NCHW, output_dtype=types.UINT8)

        jpegs2, _ = fn.external_source(
            source=eii_0, mode=types.EXTSOURCE_FNAME, max_width=2000, max_height=2000)
        output2 = fn.resize(jpegs2, resize_width=300, resize_height=300,
                           output_layout=types.NCHW, output_dtype=types.UINT8)
        
        external_source_pipeline_mode1.set_outputs(output, output2)

    # build the external_source_pipeline_mode1
    external_source_pipeline_mode1.build()
    # Index starting from 0
    cnt = 0
    # Dataloader
    data_loader = ROCALCustomIterator(
        external_source_pipeline_mode1, device=device)
    for i, output_list in enumerate(data_loader, 0):
        print("**************MODE 1*******************")
        print("**************", i, "*******************")
        print("**************starts*******************")
        print("\nImages:\n", output_list)
        print("**************ends*******************")
        print("**************", i, "*******************")
        for img in output_list[0]:
            cnt = cnt + 1
            image_dump(img, cnt, device=device, mode=1)
        for img in output_list[1]:
            cnt = cnt + 1
            image_dump(img, cnt, device=device, mode=0)
    ##################### MODE 1 #########################
    print("END*********************************************************************")

if __name__ == '__main__':
    main()
