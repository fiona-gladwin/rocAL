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

##
# @file external_source.py
#
# @brief File containing external source class

import amd.rocal.types as types
import rocal_pybind as b

class ExternalSource(object):
    def __init__(self):
        self._source = None
        self._mode = None
        self._is_external_source_operator = True
        self._user_given_width = None
        self._user_given_height = None
        self._reader_id = None

    def __init__(self, source, mode, width, height, reader_id):
        self._source = source
        self._mode = mode
        self._is_external_source_operator = True
        self._user_given_width = width
        self._user_given_height = height
        self._reader_id = reader_id

    def source(self):
        return self._source

    def mode(self):
        return self._mode
    
    def dims(self):
        return (self._user_given_width, self._user_given_height)

    def id(self):
        return self._reader_id

    def feed_input(self, source_data, handle, batch_size, eos):
        # Extract all data from the source
        images_list = source_data[0] if (self._mode == types.EXTSOURCE_FNAME) else []
        input_buffer = source_data[0] if (self._mode != types.EXTSOURCE_FNAME) else []
        labels_data = source_data[1] if (len(source_data) > 1) else None
        roi_height = source_data[2] if (len(source_data) > 2) else []
        roi_width = source_data[3] if (len(source_data) > 3) else []
        ROIxywh_list = []
        for i in range(batch_size):
            ROIxywh = b.ROIxywh()
            ROIxywh.x =  0
            ROIxywh.y =  0
            ROIxywh.w = roi_width[i] if len(roi_width) > 0 else 0
            ROIxywh.h = roi_height[i] if len(roi_height) > 0 else 0
            ROIxywh_list.append(ROIxywh)
        if (len(source_data) == 6 and self._mode == types.EXTSOURCE_RAW_UNCOMPRESSED):
            decoded_height = source_data[4]
            decoded_width = source_data[5]
        else:
            decoded_width, decoded_height = self._user_given_width, self._user_given_height

        kwargs_pybind = {
            "handle": handle,
            "source_input_images": images_list,
            "labels": labels_data,
            "input_batch_buffer": input_buffer,
            "roi_xywh": ROIxywh_list,
            "decoded_width": decoded_width,
            "decoded_height": decoded_height,
            "channels": 3,
            "external_source_mode": self._mode,
            "rocal_tensor_layout": types.NCHW,
            "eos": eos,
            "loader_id":self._reader_id}
        print("ARGUMENTS : ", kwargs_pybind)
        b.externalSourceFeedInput(*(kwargs_pybind.values()))




