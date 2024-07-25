import functools
import os
from io import BytesIO

import torch
import torchvision
import torchvision.transforms.v2 as transforms
import webdataset as wds


def _video_shortener(video_tensor, length):
    start = torch.randint(0, video_tensor.shape[0] - length, (1,))
    return video_tensor[start:start + length]


def select_video_extract(length=16):
    return functools.partial(_video_shortener, length=length)


def my_collate_fn(batch):
    output = {}
    for key in batch[0].keys():
        if key == 'video':
            output[key] = torch.stack([sample[key] for sample in batch])
        else:
            output[key] = [sample[key] for sample in batch]

    return output


def map_mp4(sample):
    return torchvision.io.read_video(BytesIO(sample), output_format="TCHW", pts_unit='sec')[0]


def map_txt(sample):
    return sample.decode("utf-8")


class WebVidDataset(wds.DataPipeline):
    def __init__(self, batch_size, tar_index, root_path, video_length=16, video_size=256, video_length_offset=0,
                 horizontal_flip=True, seed=None):

        self.dataset_full_path = os.path.join(root_path, f'webvid-uw-{{{tar_index}}}.tar')

        if isinstance(video_size, int):
            video_size = (video_size, video_size)

        for size in video_size:
            if size % 8 != 0:
                raise ValueError("video_size must be divisible by 8")

        self.pipeline = [
            wds.SimpleShardList('file:' + str(self.dataset_full_path), seed=seed),
            wds.shuffle(50),
            wds.split_by_node,
            wds.tarfile_to_samples(),
            wds.shuffle(100),
            wds.split_by_worker,
            wds.map_dict(
                mp4=map_mp4,
                txt=map_txt,
            ),
            wds.map_dict(
                mp4=transforms.Compose(
                    [
                        select_video_extract(length=video_length + video_length_offset),
                        transforms.Resize(size=video_size),
                        transforms.RandomCrop(size=video_size),
                        transforms.RandomHorizontalFlip() if horizontal_flip else transforms.Identity,
                    ]
                )
            ),
            wds.rename_keys(video="mp4", text='txt', keep_unselected=True),
            wds.batched(batch_size, collation_fn=my_collate_fn, partial=True)
        ]

        super().__init__(self.pipeline)

        self.batch_size = batch_size
        self.video_length = video_length
        self.video_size = video_size
