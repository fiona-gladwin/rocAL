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
import cv2
import os
import ctypes

class ROCALCOCOIterator(object):
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
        self.bs = self.loader._batch_size
        self.output_list = self.dimensions = self.torch_dtype = None
        self.display = display
        # Image id of a batch of images
        self.image_id = np.zeros(self.bs, dtype="int32")
        # Count of labels/ bboxes in a batch
        # self.bboxes_label_count = np.zeros(self.bs, dtype="int32")
        # Image sizes of a batch
        # self.img_size = np.zeros((self.bs * 2), dtype="int32")
        self.output_memory_type = self.loader._output_memory_type

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.loader.rocal_run() != 0:
            raise StopIteration
        else:
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


def draw_patches(img, idx, device, dtype, layout, bboxes):
    # image is expected as a tensor, bboxes as numpy
    import cv2
    if device == "cpu":
        image = img.detach().numpy()
    else:
        image = img.cpu().numpy()
    if dtype != types.UINT8:
        image = (image).astype('uint8')
    if layout == types.NCHW:
        image = image.transpose([1, 2, 0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    bboxes = np.reshape(bboxes, (-1, 4))

    for (l, t, r, b) in bboxes:
        loc_ = [l, t, r, b]
        color = (255, 0, 0)
        thickness = 2
        image = cv2.UMat(image).get()
        image = cv2.rectangle(image, (int(loc_[0]), int(loc_[1])), (int(
            (loc_[2])), int((loc_[3]))), color, thickness)
        cv2.imwrite("OUTPUT_FOLDER/COCO_READER/" +
                    str(idx)+"_"+"train"+".png", image)



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
    rocal_cpu = True if sys.argv[3] == "cpu" else False
    batch_size = int(sys.argv[4])
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    local_rank = 0
    world_size = 1

    single_reader_pipeline = Pipeline(
        batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=rocal_cpu)

    with single_reader_pipeline:
        jpegs, labels, bbox = fn.readers.coco_experimental(path=image_path, annotations_file=annotation_path)
        decode = fn.decoders.image_decoder_experimental(jpegs, output_type=types.RGB, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        res = fn.resize(decode, resize_width=224, resize_height=224,
                        output_layout=types.NCHW, output_dtype=types.UINT8)
        flip_coin = fn.random.coin_flip(probability=0.5)
        cmnp = fn.crop_mirror_normalize(decode,
                                        output_layout=types.NCHW,
                                        output_dtype=types.FLOAT,
                                        crop=(224, 224),
                                        mirror=flip_coin,
                                        mean=[0.485 * 255, 0.456 *
                                              255, 0.406 * 255],
                                        std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        single_reader_pipeline.set_outputs(cmnp, labels, bbox)

# There are 2 ways to get the outputs from the pipeline
# 1. Use the iterator
# 2. use the pipe.run()

# Method 1
    # use the iterator
    single_reader_pipeline.build()
    imageIteratorPipeline = ROCALCOCOIterator(
        single_reader_pipeline)
    cnt = 0
    for i, it in enumerate(imageIteratorPipeline):
        print(it)
        print("************************************** i *************************************", i)
        for i, img in enumerate(it[0]):
            print(img.shape)
            cnt += 1
            draw_patches(img, cnt, device=rocal_cpu, dtype=types.FLOAT, layout=types.NCHW, bboxes=it[2][i])
    imageIteratorPipeline.reset()
    print("END*********************************************************************")

# Method 2
    # iter = 0
    # # use pipe.run() call
    # output_data_batch = image_classification_train_pipeline.run()
    # print("\n Output Data Batch: ", output_data_batch)
    # # length depends on the number of augmentations
    # for i in range(len(output_data_batch)):
    #     print("\n Output Layout: ", output_data_batch[i].layout())
    #     print("\n Output Dtype: ", output_data_batch[i].dtype())
    #     for image_counter in range(output_data_batch[i].batch_size()):
    #         image = output_data_batch[i].at(image_counter)
    #         image = image.transpose([1, 2, 0])
    #         cv2.imwrite("output_images_iter" + str(i) + str(image_counter) +
    #                     ".jpg", cv2.cvtColor(image * 255, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
