"""Bounded JPEG tensor cache for the pinned SAM2 frame-loader interface.

No tracking state or history is discarded. Only decoded input images are evicted.
The float64 -> float32 conversion precedes normalization, matching the eager
loader's assignment into a preallocated float32 tensor exactly.
"""
from collections import OrderedDict
from pathlib import Path


class BoundedVideoFrames:
    def __init__(self, folder, image_size, img_mean, img_std, capacity=4):
        import torch
        self.paths = sorted((p for p in Path(folder).iterdir() if p.suffix.lower() in ['.jpg', '.jpeg']), key=lambda p: int(p.stem))
        if not self.paths:
            raise RuntimeError(f'No JPEG frames in {folder}')
        if capacity < 1:
            raise ValueError('Frame cache capacity must be positive')
        self.image_size, self.capacity = image_size, capacity
        self.mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        self.cache = OrderedDict()
        self.loads, self.hits, self.peak_cached_frames = 0, 0, 0
        self.video_height = self.video_width = None
        self[0]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        import operator
        import torch
        from sam2.utils.misc import _load_img_as_tensor
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index in self.cache:
            self.hits += 1
            self.cache.move_to_end(index)
            return self.cache[index]
        image, height, width = _load_img_as_tensor(str(self.paths[index]), self.image_size)
        if self.video_height is None:
            self.video_height, self.video_width = height, width
        if (height, width) != (self.video_height, self.video_width):
            raise ValueError('Frame dimensions changed inside one sequence')
        # Eager SAM2 casts to float32 on assignment before applying normalization.
        image = image.to(dtype=torch.float32).contiguous()
        image.sub_(self.mean).div_(self.std)
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[index] = image
        self.loads += 1
        self.peak_cached_frames = max(self.peak_cached_frames, len(self.cache))
        return image

    def statistics(self):
        return {'frames': len(self), 'decoded_loads': self.loads, 'cache_hits': self.hits,
                'capacity': self.capacity, 'peak_cached_frames': self.peak_cached_frames,
                'cached_tensor_bytes_upper_bound': self.capacity*3*self.image_size*self.image_size*4,
                'eager_tensor_bytes': len(self)*3*self.image_size*self.image_size*4}


def install(capacity=4):
    """Patch only this process; return restoration callback and instantiated caches."""
    import sam2.sam2_video_predictor as predictor_module
    original = predictor_module.load_video_frames
    instances = []

    def loader(video_path, image_size, offload_video_to_cpu,
               img_mean=(.485, .456, .406), img_std=(.229, .224, .225),
               async_loading_frames=False, compute_device=None):
        if not offload_video_to_cpu or not isinstance(video_path, str) or not Path(video_path).is_dir():
            return original(video_path=video_path, image_size=image_size,
                            offload_video_to_cpu=offload_video_to_cpu, img_mean=img_mean,
                            img_std=img_std, async_loading_frames=async_loading_frames,
                            compute_device=compute_device)
        images = BoundedVideoFrames(video_path, image_size, img_mean, img_std, capacity)
        instances.append(images)
        return images, images.video_height, images.video_width

    predictor_module.load_video_frames = loader

    def restore():
        predictor_module.load_video_frames = original

    return restore, instances
