import torch
import torch.nn as nn
import torch.nn.functional as F
from . import mix_transformer
from timm.models import create_model
import copy


class TransWCD_dual(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None, backbone_arg=None, loss="cross_entropy"):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride
        self.loss_name = loss

        if "mit" in backbone:
            self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
            self.in_channels = self.encoder.embed_dims
        elif "fastvit" in backbone:
            self.encoder = create_model(
                backbone,
                fork_feat=True,
                pretrained=pretrained,
                num_classes=self.num_classes,
                drop_rate=backbone_arg.drop,
                drop_path_rate=backbone_arg.drop_path,
                inference_mode=backbone_arg.use_inference_mode,
            )
            self.in_channels = tuple(self.embedding_dim)

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # initilize encoder
        if pretrained and "mit" in backbone:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes-1, kernel_size=1,
                                    bias=False)

        self.dropout = nn.Dropout2d(0.1)
        self.linear_pred = nn.Conv2d(self.in_channels[3], self.num_classes, kernel_size=1)


        # Difference Modules
        self.diff_c4 = conv_diff_d(in_channels=2 * c4_in_channels, out_channels=c4_in_channels)

    def get_param_groups(self):

        param_groups = [[], [], []]

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)

        return param_groups

    def forward(self, x1, x2, cam_only=False,):
        _x1 = self.encoder(x1)
        _x2 = self.encoder(x2)

        _, _, _, _c4_1 = _x1
        _, _, _, _c4_2 = _x2

        ### difference module ###
        _c4 = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))

        if cam_only:
            cam_s4 = F.conv2d(_c4, self.classifier.weight).detach()
            return cam_s4

        if self.loss_name == "cross_entropy":
            cls_x4 = self.pooling(_c4, (1, 1))
            cls_x4 = self.classifier(cls_x4)
            cls_x4 = cls_x4.view(-1, self.num_classes-1)

            return cls_x4
        
        elif self.loss_name == "contrastive" or self.loss_name == "triplet":
            # Pool the embeddings
            # Shape: [8, 512, 16, 16] -> [8, 512, 1, 1] -> [8, 512, 1] -> [8, 512]
            embeddings1_pooled = F.adaptive_avg_pool2d(_c4_1, (1, 1)).squeeze(-1).squeeze(-1)  
            embeddings2_pooled = F.adaptive_avg_pool2d(_c4_2, (1, 1)).squeeze(-1).squeeze(-1)
            
            return embeddings1_pooled, embeddings2_pooled


class TransWCD_single(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None, backbone_arg=None, loss="cross_entropy"):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride
        self.loss_name = loss

        if "mit" in backbone:
            self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
            self.in_channels = self.encoder.embed_dims
        elif "fastvit" in backbone:
            self.encoder = create_model(
                backbone,
                fork_feat=True,
                pretrained=pretrained,
                num_classes=self.num_classes,
                drop_rate=backbone_arg.drop,
                drop_path_rate=backbone_arg.drop_path,
                inference_mode=backbone_arg.use_inference_mode,
            )
            self.in_channels = tuple(self.embedding_dim)

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # initilize encoder
        if pretrained and "mit" in backbone:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes-1, kernel_size=1,
                                    bias=False)

        self.dropout = nn.Dropout2d(0.1)
        #self.linear_pred = nn.Conv2d(self.in_channels[3], self.num_classes, kernel_size=1)


        # Difference Modules
        self.diff_c4 = conv_diff_s(in_channels=2 * c4_in_channels, out_channels=c4_in_channels)

    def get_param_groups(self):

        param_groups = [[], [], []]

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)


        return param_groups

    def forward(self, x1, x2, cam_only=False):
        ### difference module ###
        # diff(A,B)
        # x = self.diff(torch.cat((x1, x2), dim=1))    #nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # _x = self.encoder(x)

        #  A-B
        x1 = x1[:, :, :, :]
        x2 = x2[:, :, :, :]
        x = torch.absolute(x1 - x2)
        _x = self.encoder(x)

        _, _, _, _c4 = _x

        if cam_only:
            cam_s4 = F.conv2d(_c4, self.classifier.weight).detach()
            return cam_s4

        assert self.loss_name == "cross_entropy"

        cls_x4 = self.pooling(_c4, (1, 1))
        cls_x4 = self.classifier(cls_x4)
        cls_x4 = cls_x4.view(-1, self.num_classes-1)

        return cls_x4


if __name__ == "__main__":
    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    transwcd = TransWCD_dual('mit_b1', num_classes=2, embedding_dim=256, pretrained=True)
    transwcd._param_groups()
    dummy_input = torch.rand(2, 3, 256, 256)
    transwcd(dummy_input)


# conv 3*3
def conv_diff_d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )

# conv 1*1 w/o relu
def conv_diff_s(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        #nn.ReLU(),
    )


def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    Args:
        model: MobileOne model in train mode.

    Returns:
        MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model
