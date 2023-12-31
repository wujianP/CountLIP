import ast
import json
import logging
import math
import os
import random
import sys
import inflect
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torchvision.transforms as trans
import torch
import torchvision.datasets as datasets
import wandb
import webdataset as wds
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist), \
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    # logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
            self,
            urls,
            weights=None,
            nshards=sys.maxsize,
            worker_seed=None,
            deterministic=False,
            epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights), \
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):

    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            if args.train_num_samples == -1:
                # adaptively compute the virtual dataset size
                num_samples = args.batch_size * args.world_size * args.steps_per_epoch
                assert num_samples <= 25870000
            else:
                num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        assert args.train_data_upsampling_factors is None, \
            "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


# >>> start: added by wjpeng >>>
class CountDataset(Dataset):
    def __init__(self, data_root, background_root, hard_num, transform, empty_fill_type, segmented, tokenizer):
        """
        This is designed for ImageNet-Boxes
        :param data_root: the root path of the dataset, default: /dev/shm/imagenet
        :param background_root: the root path of the background dataset
        :param hard_num: the number of hard negatives per sample
        :param transform: image transformation
        :param empty_fill_type: what value to fill in the empty region
        :param segmented: if true, the background of an object is removed
        """
        self.data_root = data_root
        self.background_root = background_root
        self.hard_num = hard_num
        self.transform = transform
        self.tokenize = tokenizer
        self.empty_fill_type = empty_fill_type
        self.segmented = segmented
        self.inflector = inflect.engine()
        # imagenet class id to class name, eg: 'n01440764' -> ['tench', 'Tinca tinca']
        self.id2class = {}
        with open(os.path.join(data_root, 'id2class.txt'), 'r') as file:
            for line in file.readlines():
                classId = line.strip()[:9]
                className = line.strip()[10:].split(', ')
                self.id2class[classId] = className

        # imagenet file list, ['n02791124_6215.JPEG', ..., 'n02791124_9967.JPEG', ...]
        self.image_name_list = []
        for class_id in os.listdir(os.path.join(data_root, 'boxes')):
            for img_name in os.listdir(os.path.join(data_root, 'boxes', class_id)):
                self.image_name_list.append(img_name.split('.')[0])

        self.background_images = os.listdir(self.background_root)

    def __len__(self):
        return len(self.image_name_list)

    def splice_image(self, obj_region, obj_num, bg_image, keep_ratio=False):
        # divide the whole image into (grid_size x grid_size) grid, each cell with size cell_width
        grid_size = math.ceil(math.sqrt(obj_num))
        # FIXME: depends on the input resolution of your model, 224 for CLIP here
        cell_width = math.floor(224 / grid_size)

        # resize the input object region
        w, h = obj_region.size
        if keep_ratio:
            obj_resize_ratio = cell_width / max(w, h)
            obj_w, obj_h = obj_resize_ratio * w, obj_resize_ratio * h
            obj_w, obj_h = int(obj_w), int(obj_h)
        else:
            obj_w, obj_h = cell_width, cell_width
        obj_region = obj_region.resize((obj_w, obj_h))

        # Create a 224x224 black canvas
        if self.empty_fill_type == 'white':
            canvas = Image.new('RGB', (224, 224), 'white')
        elif self.empty_fill_type == 'black':
            canvas = Image.new('RGB', (224, 224), 0)
        elif self.empty_fill_type == 'mean':
            np_img = np.array(obj_region)
            mean_r = int(np.mean(np_img[:, :, 0]))
            mean_g = int(np.mean(np_img[:, :, 1]))
            mean_b = int(np.mean(np_img[:, :, 2]))
            canvas = Image.new('RGB', (224, 224), (mean_r, mean_g, mean_b))
        elif self.empty_fill_type == 'gaussian':
            random_pixels = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
            canvas = Image.fromarray(random_pixels)
        elif self.empty_fill_type == 'real':
            canvas = Image.new('RGB', (224, 224), 0)
        else:
            raise KeyError()

        # Randomly select obj_num cells
        selected_cells = random.sample(range(grid_size * grid_size), obj_num)

        # Fill the selected cells
        for cell in selected_cells:
            row = cell // grid_size
            col = cell % grid_size
            x1 = col * cell_width
            y1 = row * cell_width
            x2 = x1 + obj_w
            y2 = y1 + obj_h
            fill_region = (x1, y1, x2, y2)
            canvas.paste(obj_region, fill_region)

        if self.empty_fill_type == 'real':
            foreground_array = np.array(canvas)
            background_array = np.array(bg_image)
            # 创建一个掩码，将前景图中黑色部分设为True，其余部分设为False
            mask = (foreground_array[:, :, 0] <= 12) & (foreground_array[:, :, 1] <= 12) & (
                    foreground_array[:, :, 2] <= 12)
            # 将前景图中黑色部分替换为背景图对应位置的像素
            combined_array = np.where(mask[:, :, np.newaxis], background_array, foreground_array)
            # 创建新的PIL图像对象
            canvas = Image.fromarray(combined_array)
        return canvas

    def __getitem__(self, idx):
        image_name = self.image_name_list[idx]
        class_id = image_name.split('_')[0]

        class_name = random.choice(self.id2class[class_id])

        image_path = os.path.join(self.data_root, 'train', class_id, image_name) + '.JPEG'
        image = Image.open(image_path).convert('RGB')

        if self.segmented:
            object_region_path = os.path.join(self.data_root, 'segments', class_id, image_name) + '.jpg'
            object_region = Image.open(object_region_path).convert('RGB')
        else:
            box_path = os.path.join(self.data_root, 'boxes', class_id, image_name) + '.xml'
            box_ann = ET.parse(box_path).getroot()
            x_min = int(box_ann.find('object').find('bndbox').find('xmin').text)
            y_min = int(box_ann.find('object').find('bndbox').find('ymin').text)
            x_max = int(box_ann.find('object').find('bndbox').find('xmax').text)
            y_max = int(box_ann.find('object').find('bndbox').find('ymax').text)
            object_region = image.crop((x_min, y_min, x_max, y_max))

        # generate n spliced images
        object_nums = random.sample(range(1, 11), self.hard_num)
        bg_path = random.choice(self.background_images)
        bg_img = Image.open(os.path.join(self.background_root, bg_path))
        w, h = bg_img.size
        side_length = min(w, h)
        # (left, upper, right, lower)
        left, top, right, bottom = (w - side_length) / 2, (h - side_length) / 2, (w + side_length) / 2, (
                    h + side_length) / 2
        bg_img = bg_img.crop((left, top, right, bottom)).resize((224, 224))
        images = []
        texts = []

        for i in range(self.hard_num):
            object_num = object_nums[i]
            spliced_img = self.splice_image(obj_region=object_region, obj_num=object_num, bg_image=bg_img)
            if self.transform is not None:
                spliced_img = self.transform(spliced_img)
            images.append(spliced_img)
            texts.append(f'a photo of {self.inflector.number_to_words(object_num)} {self.inflector.plural(class_name)}')

        if self.tokenize is not None:
            texts = self.tokenize(texts)
            texts = [texts[i] for i in range(self.hard_num)]

        # convert two list into a tuple (hard-img-1, ..., hard-img-n, text-1, ..., text-n)
        ret = tuple(images + texts)

        return ret


def get_count_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    # FIXME: check this, this is only useful for CLIP model
    transform_fn = trans.Compose([
        trans.Resize((224, 224)),
        preprocess_fn.transforms[-3],
        preprocess_fn.transforms[-2],
        preprocess_fn.transforms[-1]
    ])
    dataset = CountDataset(data_root=args.count_data_root,
                           background_root=args.count_background_root,
                           hard_num=args.hard_num,
                           transform=transform_fn,
                           empty_fill_type=args.empty_fill_type,
                           segmented=args.segmented_object,
                           tokenizer=tokenizer)

    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.count_batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class GoogleCountBench(Dataset):
    def __init__(self, data_root, transform, tokenizer):
        self.data_root = data_root
        self.transform = transform
        self.tokenize = tokenizer

        files = os.listdir(self.data_root)
        self.image_list = []
        self.ann_list = []
        for file in files:
            if file.endswith('.jpg'):
                self.image_list.append(file)
            if file.endswith('.json'):
                self.ann_list.append(file)
        self.image_list = sorted(self.image_list)
        self.ann_list = sorted(self.ann_list)

        # check correctness
        assert len(self.ann_list) == len(self.image_list)
        for img, ann in zip(self.image_list, self.ann_list):
            assert img.split('.')[0] == ann.split('.')[0]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.data_root, image_name)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        ann_name = self.ann_list[idx]
        ann_path = os.path.join(self.data_root, ann_name)
        with open(ann_path, 'r') as file:
            ann = json.load(file)
        file.close()

        all_texts = self.generate_all_text(ann)
        all_texts = self.tokenize(all_texts)

        label = ann['number'] - 2  # convert number to the index in all_texts

        return image, all_texts, label

    @staticmethod
    def generate_all_text(ann):
        origin_text = ann['text'].lower()
        number_word = ann['number_word'].lower()
        p = inflect.engine()
        all_number_words = [p.number_to_words(num) for num in range(2, 11)]
        all_texts = [origin_text.replace(number_word, new_num_word) for new_num_word in all_number_words]

        return all_texts


# <<< end: added by wjpeng <<<


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == 'count':
        return get_count_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    #  FIXME: only train mode supported
    if args.dataset_type == 'mix':
        """both normal text-image dataset and counting dataset"""
        data["train-normal"] = get_wds_dataset(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer
        )
        data['train-count'] = get_count_dataset(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer
        )
        google_dataset = GoogleCountBench(data_root=args.google_val_data_root,
                                          transform=preprocess_val,
                                          tokenizer=tokenizer)
        data['google-count'] = DataLoader(
            dataset=google_dataset,
            batch_size=64,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )
        # data iterator
        data['iterator'] = {
            'train-normal': iter(data["train-normal"].dataloader),
            'train-count': iter(data["train-count"].dataloader)
        }

    else:
        if args.train_data or args.dataset_type == "synthetic":
            data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
                args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

        if args.val_data:
            data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
                args, preprocess_val, is_train=False, tokenizer=tokenizer)

        if args.imagenet_val is not None:
            data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

        if args.imagenet_v2 is not None:
            data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
