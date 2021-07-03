import yaml
import torch
import torch.nn as nn
from . import layers

class Model(nn.Module):

    def __init__(self, yaml_file):
        super(Model, self).__init__()

        with open(yaml_file, 'r') as file:
            model_cfg = yaml.load(file.read(), Loader=yaml.FullLoader)

        self.layers  = []
        self.sources = []
        self.nums    = []
        self.input_indexs = set()
        layers_config = model_cfg['layers']
        for n, line in enumerate(layers_config):
            sources, layer_name, args, kwargs, num = line

            if isinstance(sources, int):
                sources = [sources]

            if not isinstance(num, int) or num <= 0:
                assert False, "layer's num must be int and > 0"

            self.layers.append(
                eval(f"layers.{layer_name}")(*args, **kwargs)
            )

            indexs = []
            for source in sources:
                if source < 0:
                    index = len(self.sources) + source
                    assert index >= 0, "找不到输入层"
                    indexs.append(index)
                else:
                    self.input_indexs.add(n)
                    indexs.append(-(source + 1))
            self.sources.append(indexs)
            self.nums.append(num)

        # get output layers index
        all_indexs = set()
        index_been_used = set()
        for i, indexs in enumerate(self.sources):
            all_indexs.add(i)
            for index in indexs:
                index_been_used.add(index)

        self.output_indexs = all_indexs - index_been_used
        self.layers = nn.Sequential(*self.layers)


    def get_layer_output(self, index, forward_dict):
        if index in forward_dict.keys():
            return forward_dict[index]
        else:
            source_outputs = []
            for source_index in self.sources[index]:
                source_outputs.append(self.get_layer_output(source_index, forward_dict))

            output = self.layers[index](*source_outputs)
            forward_dict[index] = output
            return output

    def forward(self, *inputs, **kwargs):
        assert len(inputs) == len(self.input_indexs), ""
        forward_dict = {}
        for i, input in enumerate(inputs):
            forward_dict[-(i + 1)] = input

        outputs = [self.get_layer_output(output_index, forward_dict) for output_index in self.output_indexs]

        if len(outputs) == 1:
            return outputs[0]

        return outputs


if __name__ == "__main__":
    model = Model('./tmp.yaml')

    input = torch.zeros((32, 3, 112, 96))
    output = model(input)

    if isinstance(output, list):
        for o in output:
            print(o.shape)
    else:
        print(output.shape)
