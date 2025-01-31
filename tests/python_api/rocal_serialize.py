# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
from amd.rocal.plugin.pytorch import ROCALClassificationIterator

from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import sys
import cv2
import os

def main():
    if len(sys.argv) < 3:
        print('Please pass image_folder cpu/gpu batch_size')
        exit(0)
    try:
        path = "output_folder/file_reader/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    data_path = sys.argv[1]
    rocal_cpu = True if sys.argv[2] == "cpu" else False
    batch_size = int(sys.argv[3])
    num_threads = 1
    device_id = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    local_rank = 0
    world_size = 1

    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=rocal_cpu)

    with pipe:
        jpegs, _ = fn.readers.file(file_root=data_path)
        decode = fn.decoders.image(jpegs, file_root=data_path, shard_id=local_rank, num_shards=world_size,)
        res = fn.resize(decode, resize_width=224, resize_height=224,
                        output_layout=types.NHWC, output_dtype=types.UINT8)
        # flip_coin = fn.random.coin_flip(probability=0.5)
        # cmnp = fn.crop_mirror_normalize(res,
        #                                 output_layout=types.NCHW,
        #                                 output_dtype=types.FLOAT,
        #                                 crop=(224, 224),
        #                                 mirror=1,
        #                                 mean=[0.485 * 255, 0.456 *
        #                                       255, 0.406 * 255],
        #                                 std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        pipe.set_outputs(res)

    pipe.build()
    serialized_string = pipe.serialize()    # Serialize the pipeline
    print("Serialize : ", serialized_string)
    
    new_deserialized_pipe = Pipeline.deserialize(serialized_string) # Deserialize the pipeline from the serialized string
    new_deserialized_pipe.build()

    output_data_batch_new = new_deserialized_pipe.run() # Run deserialized pipeline and fetch outputs
    output_data_batch = pipe.run()

    if len(output_data_batch_new) != len(output_data_batch):
        print("The outputs of the serialized and deserialized pipeline are different")
        exit()
    if len(output_data_batch_new) > 0:
        if output_data_batch_new[0].batch_size() != output_data_batch[0].batch_size():
            print("The batch size of the serialized and deserialized pipeline are different")
            exit()
        
    for i in range(len(output_data_batch_new)):
        print("\n Output Layout: ", output_data_batch_new[i].layout())
        print("\n Output Dtype: ", output_data_batch_new[i].dtype())
        print("Output Batch size : ", output_data_batch_new[i].batch_size())
        for image_counter in range(output_data_batch_new[i].batch_size()):
            image1 = output_data_batch_new[i].at(image_counter)
            image2 = output_data_batch[i].at(image_counter)
            if (image1 == image2).all():
                print("SUCCESS : The outputs of serialized and deserialized pipeline are the same")
            # image1 = image1.transpose([1, 2, 0])
            # image2 = image2.transpose([1, 2, 0])
            cv2.imwrite("output_images_new_" + str(i) + str(image_counter) +
                        ".jpg", cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))
            cv2.imwrite("output_images_" + str(i) + str(image_counter) +
                        ".jpg", cv2.cvtColor(image2, cv2.COLOR_RGB2BGR))
    

if __name__ == '__main__':
    main()
