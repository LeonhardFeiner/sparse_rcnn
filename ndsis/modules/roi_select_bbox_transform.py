import torch
import torch.nn as nn
from ndsis.utils.bbox import clip_boxes, round_bbox, clip_boxes_asymmetric


class Adder(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, bbox_tensor):
        return bbox_tensor + bbox_tensor.new_tensor(self.value)


class Divider(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, bbox_tensor):
        return bbox_tensor / bbox_tensor.new_tensor(self.value)


class AsymmetricRounder(nn.Module):
    def forward(self, bbox_tensor):
        return round_bbox(bbox_tensor)


class AsymmetricClipper(nn.Module):
    def forward(self, bbox_tensor, shape):
        return clip_boxes_asymmetric(
            bbox_tensor, bbox_tensor.new_tensor(tuple(shape)))


class OffsetClipper(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset

    def forward(self, bbox_tensor, shape):
        max_coords = bbox_tensor.new_tensor(shape) + self.offset
        return clip_boxes(bbox_tensor, max_coords)


class IdentityOfFirst(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class BboxTransformer(nn.Module):
    def __init__(self, inner_sequence, clipper=None):
        super().__init__()
        self.inner_sequence = inner_sequence
        self.clipper = clipper or IdentityOfFirst()

    def forward(self, bbox_batch, shape=None):
        box_sample_count = [
            len(bbox_sample) for bbox_sample in bbox_batch]

        raw_bbox_tensor = torch.cat(bbox_batch)

        bbox_tensor = self.inner_sequence(raw_bbox_tensor)

        clipped_bbox_tensor = self.clipper(bbox_tensor, shape)

        bbox_sample_association = torch.cat([
            torch.full((bbox_count,), index, dtype=torch.long)
            for index, bbox_count in enumerate(box_sample_count)])

        return clipped_bbox_tensor, box_sample_count, bbox_sample_association


class BBoxTransformerInterpolation(BboxTransformer):
    def __init__(self, clip=True, resize=None):
        pixel_offset = -0.5
        shape_offset = -1

        raw_sequence = list()
        if resize is not None:
            raw_sequence.append(Divider(resize))
        raw_sequence.append(Adder(pixel_offset))
        inner_sequence = nn.Sequential(*raw_sequence)
        clipper = clip and OffsetClipper(shape_offset)
        super().__init__(inner_sequence, clipper)


class BBoxTransformerSlice(BboxTransformer):
    def __init__(self, clip=False, resize=None):

        raw_sequence = list()
        if resize is not None:
            raw_sequence.append(Divider(resize))
        raw_sequence.append(AsymmetricRounder())
        inner_sequence = nn.Sequential(*raw_sequence)
        clipper = clip and AsymmetricClipper()

        super().__init__(inner_sequence, clipper)


class BBoxTransformerExample(BboxTransformer):
    def __init__(self, clip=False, resize=None):
        pixel_offset = -0.5
        shape_offset = -1
        quantize = False

        raw_sequence = list()

        if resize is not None:
            raw_sequence.append(Divider(resize))

        raw_sequence.append(Adder(pixel_offset))

        if quantize:
            raw_sequence.append(AsymmetricRounder())

        inner_sequence = nn.Sequence(raw_sequence)

        clipper = OffsetClipper(shape_offset) if clip else None

        super().__init__(inner_sequence, clipper)
