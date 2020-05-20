from typing import Iterable, List, Union, Any, Tuple
import torch
from torch.nn.functional import pad
import numpy as np


def cummax(
        tensor: torch.tensor, dim: int = 0) -> torch.tensor:
    """Return the cumulative maximum of the elements along a given axis.

    Args:
        tensor (torch.tensor): input tensor
        dim (int, optional): Defaults to 0.
            Axis along which the cumulative maximum is computed

    Returns:
        max_tensor (torch.tensor):
        A tensor with the same size as the input holding the cumulative
            maximum of the input tensor.
    """
    ret_val = tensor.clone()
    ret_val_unbind = ret_val.unbind(dim)
    if len(ret_val_unbind):
        for current, last in zip(ret_val_unbind[1:], ret_val_unbind):
            current[()] = torch.max(current, last)
    return ret_val


def cummin(
        tensor: torch.tensor, dim: int = 0) -> torch.tensor:
    """Return the cumulative minimum of the elements along a given axis.

    Args:
        tensor (torch.tensor): input tensor
        dim (int, optional): Defaults to 0.
            Axis along which the cumulative minimum is computed

    Returns:
        max_tensor (torch.tensor):
        A tensor with the same size as the input holding the cumulative
            minimum of the input tensor.
    """
    ret_val = tensor.clone()
    ret_val_unbind = ret_val.unbind(dim)
    if len(ret_val_unbind):
        for current, last in zip(ret_val_unbind[1:], ret_val_unbind):
            current[()] = torch.min(current, last)
    return ret_val


def cummax_reversed(
        tensor: torch.tensor, dim: int = 0) -> torch.tensor:
    """Return the cumulative maximum of the elements along a given axis in
        inverse order.

    Args:
        tensor (torch.tensor): input tensor
        dim (int, optional): Defaults to 0.
            Axis along which the cumulative maximum is computed in inverse
            order

    Returns:
        max_tensor (torch.tensor):
        A tensor with the same size as the input holding the cumulative
            maximum of the input tensor.
    """
    return cummax(tensor.flip(dim), dim).flip(dim)


def cummin_reversed(
        tensor: torch.tensor, dim: int = 0) -> torch.tensor:
    """Return the cumulative minimum of the elements along a given axis in
        inverse order.

    Args:
        tensor (torch.tensor): input tensor
        dim (int, optional): Defaults to 0.
            Axis along which the cumulative minimum is computed in inverse
            order

    Returns:
        max_tensor (torch.tensor):
        A tensor with the same size as the input holding the cumulative
            minimum of the input tensor.
    """
    return cummin(tensor.flip(dim), dim).flip(dim)


def product(elements: Iterable[Any]) -> Any:
    """Calculates the product of a iterable of elements.

    Args:
        elements (Iterable[Any]): an iterable of elements to multiply

    Returns:
        product [Any]: the product of the elements.
        Defaults to 1 if elements is empty.
    """
    result = 1
    for elem in elements:
        result = result * elem
    return result


def zip_select(
        data_iterable: Iterable,
        indicator_iterable: Iterable) -> List:
    """Zips the data_iterable and the indicator_iterable to selects from
        the data samples using the indicator samples.

    Args:
        data_iterable (Iterable): a iterable containing data samples which can
            be indicated.
        indicator_iterable (Iterable): a iterable containing indicator samples
            which can indicate data samples.
    Returns:
        selected_list (List): a list of the selected data
    """
    return [
        sample_data[sample_indicator]
        for sample_data, sample_indicator
        in zip(data_iterable, indicator_iterable)]


def default_argmax(
        tensor: torch.tensor,
        dimension: int) -> torch.LongTensor:
    """Workaround for torch.argmax to enable empty tensors in case the given
        dimension is not of size zero.

    Returns the indices of the maximum value of all elements in the input
    tensor.
    Workaround for https://github.com/pytorch/pytorch/issues/28380

    Args:
        tensor (torch.tensor): the input tensor.
        dimension (int):  the dimension to reduce.

    Returns:
        torch.LongTensor: 
            the indices of the maximum values of a tensor across a dimension.
    """
    if tensor.shape[dimension] != 0 and tensor.numel() == 0:
        new_shape = list(tensor.shape)
        del new_shape[dimension]
        return tensor.new_zeros(new_shape)
    else:
        return tensor.argmax(dimension)


def default_argmin(
        tensor: torch.tensor,
        dimension: int) -> torch.LongTensor:
    """Workaround for torch.argmin to enable empty tensors in case the given
        dimension is not of size zero.

    Returns the indices of the minimum value of all elements in the input
    tensor.
    Workaround for https://github.com/pytorch/pytorch/issues/28380

    Args:
        tensor (torch.tensor): the input tensor.
        dimension (int):  the dimension to reduce.

    Returns:
        torch.LongTensor: 
            the indices of the minimum values of a tensor across a dimension.
    """
    if tensor.shape[dimension] != 0 and tensor.numel() == 0:
        new_shape = list(tensor.shape)
        del new_shape[dimension]
        return tensor.new_zeros(new_shape)
    else:
        return tensor.argmin(dimension)


def split_select_nd(
        tensor: torch.tensor,
        splits_dims: Union[torch.tensor, List[torch.tensor]],
        dims: List[int] = None) -> List[torch.tensor]:
    """splits the tensor into a squared table of tensors and selects the
        tensors lying on the diagonal of the table.

    Args:
        tensor (torch.tensor): the tensor to split and select
        splits_dims (Union[torch.tensor, List[torch.tensor]]):
            a retangular table of splits. every column has to sum to the
            length of the associated dimensions of the tensor.
        dims (List[int]): the dimensions of the tensor which are associated by
            a column of split_dims. The length has to be equal to the length of
            split dims. dims must be valid when indexing tensor.shape[dim]

    Returns:
        selected_tensors (List[torch.tensor]): [description]
    """
    if dims is None:
        dims = range(len(splits_dims))
    else:
        assert len(dims) == len(splits_dims)
    if isinstance(splits_dims, torch.Tensor):
        split_dims_transposed = splits_dims.T
    else:
        split_dims_transposed = torch.stack(splits_dims, dim=-1)

    assert split_dims_transposed.ndim == 2
    assert all(
        tensor.shape[dim] == split_sum_dim
        for dim, split_sum_dim in zip(dims, split_dims_transposed.sum(0)))

    stop_positions = split_dims_transposed.cumsum(0)
    start_stop_positions = torch.stack(
        (pad(stop_positions, (0, 0, 1, -1)), stop_positions), dim=-1)

    return [
        tensor[current_slice]
        for current_slice in slice_tuple_gen(dims, start_stop_positions)]


def slice_tuple_gen(
        dims: List[int],
        start_stop_splits: Iterable[torch.tensor]) -> Iterable[Tuple[slice]]:
    """Generates a tuple of slices to slice a tensor multiple times.

    Args:
        dims (List[int]): the list of dimensions which are associated to
            columns of start_stop_splits
        start_stop_splits (Iterable[torch.tensor]):
            a list of start and stop positions to be ordered into an tuple
            of slices. the certain start and stop positions are put into the
            tuple index given by the associated dim

    Yields:
        Tuple[slice, ...]: Tuples which can be used to slice a tensor
    """
    if any(dim < 0 for dim in dims):
        ellipsis_pos = max(-1, *dims) + 1
        array_len = ellipsis_pos + 1 - min(dims)
        array = [slice(None)] * array_len
        array[ellipsis_pos] = Ellipsis
    else:
        array = [slice(None)] * (max(dims) + 1)

    for start_stop_combination in start_stop_splits:
        for dim, (start, stop) in zip(dims, start_stop_combination):
            array[dim] = slice(start, stop)
        yield tuple(array)


def get_slice_gen(
        dims: List[int],
        start_stop_splits: Iterable[torch.tensor],
        ndims: int) -> Iterable[Tuple[slice]]:
    slice_array = np.full((len(start_stop_splits), ndims), slice(None))

    slice_array[: dims] = [
        [slice(start, stop) for start, stop in split]
        for split in start_stop_splits]


def slice_along_axis(arr, indices, axis):
    slice_array = [slice(None)] * arr.ndim
    slice_array[axis] = indices
    return arr[tuple(slice_array)]


def is_sorted(tensor: torch.tensor, axis=0) -> torch.BoolTensor:
    return (
        slice_along_axis(tensor, slice(-1), axis) <= 
        slice_along_axis(tensor, slice(1, None), axis)).all()
#    return (tensor[:-1] <= tensor[1:]).all()


def split_list(the_list, splits):
    assert len(the_list) == sum(splits), (len(the_list), sum(splits))
    cum_splits = np.cumsum([0, *splits])
    return [
        the_list[start: stop]
        for start, stop
        in zip(cum_splits, cum_splits[1:])]
