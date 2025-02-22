import torch as th
import yaml as ym
from math import log2
from torch.nn import (
    Linear,
    ReLU,
    Tanh,
    Dropout,
    LayerNorm,
    Module,
    Sequential,
    Sigmoid,
    Softmax,
    Conv2d,
    BatchNorm2d,
    MaxPool2d,
    Dropout2d
    
)

__activation__ = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
}
class LinearBlock(Module):

    def __init__(
        self,
        params: dict,
    ) -> None:
        
        super().__init__()
        self._net_ = Sequential(
            Linear(in_features=params["in_features"], out_features=params["out_features"]),
            LayerNorm(normalized_shape=params["out_features"]),
            Dropout(p=params["dp_rate"]),
            __activation__[params["activation"]]()
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net_(inputs)

class ConvDownBlock(Module):

    def __init__(
        self,
        params: dict
    ) -> None:
        
        super().__init__()
        self._net = Sequential(
            Conv2d(
                in_channels=params["in_channels"],
                out_channels=params["out_channels"],
                kernel_size=params["kernel_size"],
                stride=params["stride"],
                padding=params["padding"]
            ),
            Dropout2d(p=params["dp_rate"]),
            BatchNorm2d(num_features=params["out_channels"]),
            __activation__[params["activation"]]()
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net(inputs)



__modules__ = {
    "linear": LinearBlock,
    "conv2d_down": ConvDownBlock
}
# class BaseConv(Module):
    
#     def __init__(self, params: dict | str) -> None:




class BaseModule(Module):

    def __init__(self, params: dict | str) -> None:

        super().__init__()
        self.params = params
        if isinstance(params, str):
            with open(params, "r") as yaml:
                self.params = ym.load(yaml)
        
        backbone_buffer = []
        head_buffer = []
        self.__call_operators__ = {
            "conv --> lin": self._conv2lin_call_
        }
        for backbone_module in self.params["BackBone"].keys():
            params = self.params["BackBone"][backbone_module]
            backbone_buffer.append(self._build_module_(params, backbone_module))

        for head_module in self.params["Head"].keys():
            params = self.params["Head"][head_module]
            head_buffer.append(self._build_module_(params, head_module))
        
        self._backbone_ = Sequential(*backbone_buffer)
        self._head_ = Sequential(*head_buffer)

        if self.params["ModelType"] == "conv --> lin":
            
            last_block = list(self.params["BackBone"].values())[-1]
            last_module = list(last_block.values())[-1]
            layers_n = last_block["layers_n"]
            img_size = last_block["img_size"]
            out_ch = last_module["out_channels"]

            
            inf = (
                out_ch * 
                img_size[0] // (2 ** layers_n) * 
                img_size[1] // (2 ** layers_n)
            )
            ouf = list(self.params["Head"].values())[-1]["out_block"]["out_features"]
            self._lin_ = Linear(inf, ouf)
        
        
    def _build_module_(self, module_params: dict, module_name: str) -> Module:

    
        layers_buffer = []
        in_layer = __modules__[module_name](module_params["input_block"])
        out_layer = __modules__[module_name](module_params["out_block"])
        
        layers_buffer = [in_layer, out_layer]
        if (module_params["layers_n"] > 2) and ("hiden_block" in module_params.keys()):
            
            try:
                layers_buffer.pop(-1)
                hiden_layers = [
                    __modules__[module_name](module_params["hiden_block"])
                    for _ in range(module_params["layers_n"] - 2)
                ]
                layers_buffer += hiden_layers + [out_layer, ]
            
            except:
                print("!!!User Warning, hiden_block params can't be expected!!!")
        
        return Sequential(*layers_buffer)
    

    def _conv2lin_call_(self, inputs: th.Tensor) -> th.Tensor:

        conv = th.flatten(self._backbone_(inputs), start_dim=1)
        print(conv.size())
        print(self._lin_(conv).size())
        
        return self._head_(conv)
        
     
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self.__call_operators__[self.params["ModelType"]](inputs)



if __name__ == "__main__":

    params = {
        "ModelType": "conv --> lin",
        "BackBone": {
            "conv2d_down": {
                "img_size": (128, 128),
                "layers_n": 1,
                "input_block": {
                    "in_channels": 3,
                    "out_channels": 32,
                    "kernel_size": (3, 3),
                    "padding": 0,
                    "stride": 2,
                    "dp_rate": 0.45,
                    "activation": "relu"
                },
                "out_block": {
                    "in_channels": 32,
                    "out_channels": 128,
                    "kernel_size": (3, 3),
                    "padding": 0,
                    "stride": 2,
                    "dp_rate": 0.45,
                    "activation": "tanh"
                }
            }
        },
        "Head": {
            "linear": {
                "layers_n": 1,
                "input_block": {
                    "in_features": 4,
                    "out_features": 128,
                    "activation": "relu",
                    "dp_rate": 0.45 
                },
                "hiden_block": {
                    "in_features": 128,
                    "out_features": 128,
                    "activation": "relu",
                    "dp_rate": 0.45
                },
                "out_block": {
                    "in_features": 128,
                    "out_features": 32,
                    "activation": "tanh",
                    "dp_rate": 0.45
                },
            },
        },

    }
    
    inputs = th.normal(0.12, 1.12, (10, 3, 128, 128))
    model = BaseModule(params)
    print(model(inputs))


