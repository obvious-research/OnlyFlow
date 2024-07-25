import functools
from io import BytesIO

import torch
import torchvision
import torchvision.transforms.v2 as transforms
import wids
from torch.utils.data import DataLoader


def _video_shortener(video_tensor, length, generator=None):
    start = torch.randint(0, video_tensor.shape[0] - length, (1,), generator=generator)
    return video_tensor[start:start + length]


def select_video_extract(length=16, generator=None):
    return functools.partial(_video_shortener, length=length, generator=generator)


def my_collate_fn(batch):
    videos = torch.stack([sample[0] for sample in batch])
    txts = [sample[1] for sample in batch]

    return videos, txts


class WebVidDataset(wids.ShardListDataset):

    def __init__(self, shards, cache_dir, video_length=16, video_size=256, video_length_offset=1, val=False, seed=42,
                 **kwargs):

        self.val = val
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.generator_init_state = self.generator.get_state()
        super().__init__(shards, cache_dir=cache_dir, keep=True, **kwargs)

        if isinstance(video_size, int):
            video_size = (video_size, video_size)

        self.video_size = video_size

        for size in video_size:
            if size % 8 != 0:
                raise ValueError("video_size must be divisible by 8")

        self.transform = transforms.Compose(
            [
                select_video_extract(length=video_length + video_length_offset, generator=self.generator),
                transforms.Resize(size=video_size),
                transforms.RandomCrop(size=video_size) if not self.val else transforms.CenterCrop(size=video_size),
                transforms.RandomHorizontalFlip() if not self.val else transforms.Identity(),
            ]
        )

        self.add_transform(self._make_sample)

    def _make_sample(self, sample):
        if self.val:
            self.generator.set_state(self.generator_init_state)
        video = torchvision.io.read_video(BytesIO(sample[".mp4"].read()), output_format="TCHW", pts_unit='sec')[0]
        label = sample[".txt"]
        return self.transform(video), label


if __name__ == "__main__":

    dataset = WebVidDataset(
        tar_index=0,
        root_path='/users/Etu9/3711799/onlyflow/data/webvid/desc.json',
        video_length=16,
        video_size=256,
        video_length_offset=0,
    )

    sampler = wids.DistributedChunkedSampler(dataset, chunksize=1000, shuffle=True)
    dataloader = DataLoader(
        dataset,
        collate_fn=my_collate_fn,
        batch_size=4,
        sampler=sampler,
        num_workers=4
    )

    for i, (images, labels) in enumerate(dataloader):
        print(i, images.shape, labels)
        if i > 10:
            break
