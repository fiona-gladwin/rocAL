/*
Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma once

#include "video_decoder.h"
#include "roc_video_dec.h"
#include "video_demuxer.h"

#ifdef ROCAL_VIDEO

class RocDecVideoDecoder : public VideoDecoder {
    public:
        //! Default constructor
        RocDecVideoDecoder();
        VideoDecoder::Status Initialize(const char *src_filename, int device_id = 0) override;
        VideoDecoder::Status Decode(unsigned char *output_buffer, unsigned seek_frame_number, size_t sequence_length, size_t stride, int out_width, int out_height, int out_stride, AVPixelFormat out_format) override;
        int seek_frame(AVRational avg_frame_rate, AVRational time_base, unsigned frame_number) override;
        void release() override;
        ~RocDecVideoDecoder() override;

    private:
        const char *_src_filename = NULL;
        std::shared_ptr<VideoDemuxer> _demuxer;
        std::shared_ptr<RocVideoDecoder> _rocvid_decoder;

        int _codec_width, _codec_height;
};
#endif