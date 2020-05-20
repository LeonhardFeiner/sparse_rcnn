import torch
import torch.nn as nn


class SequentialInterims(nn.Sequential):
    def _inner_forward(self, x):
        for module in self._modules.values():
            x = module(x)
            yield x

    def forward(self, x):
        return list(self._inner_forward(x))


class Elementwise(nn.Module):
    def __init__(self, operation):
        super(Elementwise, self).__init__()
        self.operation = operation

    def forward(self, x_list):
        return [self.operation(x) for x in x_list]


class ZipApply(nn.Sequential):
    def forward(self, x_list):
        return [module(x) for module, x in zip(self._modules.values(), x_list)]


class InverseSequentialInterims(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.module_list = nn.ModuleList(modules)

    def forward(self, x_input, skip_list):
        assert len(self.module_list) == len(skip_list), (
            len(self.module_list), len(skip_list))
        for skip, module in zip(skip_list, self.module_list):
            x_input = module(x_input, skip)
        return x_input


class ReuniteSequentialInterims(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.module_list = nn.ModuleList(modules)

    def _inner_forward(self, x, skip_list):
        assert len(self.module_list) == len(skip_list), (
            len(self.module_list), len(skip_list))
        for skip, module in zip(skip_list, self.module_list):
            x = module(x, skip)
            yield x

    def forward(self, x, skip_list):
        return list(self._inner_forward(x, skip_list))


class UnetContainer(nn.Module):
    def __init__(self, downsampling_layer, upsampling_layer):
        super().__init__()
        self.downsampling_layer = downsampling_layer
        self.upsampling_layer = upsampling_layer

    def forward(self, x):
        *skip, out = self.downsampling_layer(x) 
        upsampled = self.upsampling_layer(out, skip[::-1])
        return upsampled


class SkipConnectionReuniter(nn.Module):
    def __init__(self, input_stage, combiner, channel_changer, output_stage):
        super().__init__()
        self.input_stage = input_stage
        self.combiner = combiner
        self.channel_changer = channel_changer
        self.output_stage = output_stage

    def forward(self, x, skip):
        x_converted = self.input_stage(x)
        combined_raw = self.combiner([x_converted, skip])
        combined = self.channel_changer(combined_raw)
        out = self.output_stage(combined)
        return out



class ModuleMap(nn.Sequential):
    def forward(self, x):
        return [module(x) for module in self._modules.values()]


class ConcatOutputListList(nn.Module):
    def __init__(self, operation):
        super(ConcatOutputListList, self).__init__()
        self.operation = operation

    def forward(self, x_input):
        x_list_list = self.operation(x_input)
        return [x for x_list in x_list_list for x in x_list]


class ConditionalStage(nn.Module):
    def __init__(self, train_module, val_module):
        super().__init__()
        self.train_module = train_module
        self.val_module = val_module

    def get_module(self):
        return (
            self.__dict__['_modules']['train_module']
            if self.__dict__['training'] else
            self.__dict__['_modules']['val_module'])

    def __getattr__(self, attr):
        module = self.get_module()
        return getattr(module, attr)

    def forward(self, *args, **kwargs):
        module = self.get_module()
        return module(*args, **kwargs)


class DefaultOutput(nn.Module):
    def __init__(self, non_empty, empty):
        super().__init__()
        self.non_empty = non_empty
        self.empty = empty

    def forward(self, x):
        return (self.non_empty if len(x) else self.empty)(x)
