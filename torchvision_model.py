import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models.resnet as resnet
import torchvision.models.mobilenet as mobilenet
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict
from anchors import Anchors
from utils import RegressionTransform
from nets.mobilenet025 import MobileNetV1
from nets.model_v3 import mobilenet_v3_large
import losses
from cfg.config import cfg_mnet
from cbam_model import ChannelAttention, SpatialAttention

class ContextModule(nn.Module):
    def __init__(self,in_channels=64):
        super(ContextModule,self).__init__()
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels)
        )
        self.det_context_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True)
        )
        self.det_context_conv2 = nn.Sequential(
            nn.Conv2d(in_channels//2, in_channels//2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels//2)
        )
        self.det_context_conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels//2, in_channels//2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True)
        )
        self.det_context_conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels//2, in_channels//2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels//2)
        )
        self.det_concat_relu = nn.ReLU(inplace=True)

        self.ca1 = ChannelAttention(in_channels)
        self.ca2 = ChannelAttention((in_channels // 2))
        self.ca3 = ChannelAttention((in_channels // 2))
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()

    def forward(self,x):
        x1 = self.det_conv1(x)
        x_ = self.det_context_conv1(x)
        x2 = self.det_context_conv2(x_)
        x3_ = self.det_context_conv3_1(x_)
        x3 = self.det_context_conv3_2(x3_)

        x1 = self.ca1(x1) * x1
        x1 = self.sa1(x1) * x1
        x2 = self.ca2(x2) * x2
        x2 = self.sa2(x2) * x2
        x3 = self.ca3(x3) * x3
        x3 = self.sa3(x3) * x3

        out = torch.cat((x1,x2,x3),1)
        act_out = self.det_concat_relu(out)

        return act_out

class FeaturePyramidNetwork(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FeaturePyramidNetwork,self).__init__()
        self.lateral_blocks = nn.ModuleList()
        self.context_blocks = nn.ModuleList()
        self.aggr_blocks = nn.ModuleList()
        for i, in_channels in enumerate(in_channels_list):
            if in_channels == 0:
                continue
            lateral_block_module = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0),    #1*1卷积
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            aggr_block_module = nn.Sequential(
                nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1),   #3*3卷积
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            context_block_module = ContextModule(out_channels)
            self.lateral_blocks.append(lateral_block_module)
            self.context_blocks.append(context_block_module)
            if i > 0 :
                self.aggr_blocks.append(aggr_block_module)

        # initialize params of fpn layers
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)                

    def forward(self,x):
        names = list(x.keys())
        x = list(x.values())
        last_inner = self.lateral_blocks[-1](x[-1])
        results = []
        results.append(self.context_blocks[-1](last_inner))
        for feature, lateral_block, context_block, aggr_block in zip(
            x[:-1][::-1], self.lateral_blocks[:-1][::-1], self.context_blocks[:-1][::-1], self.aggr_blocks[::-1]
            ):
            if not lateral_block:
                continue
            lateral_feature = lateral_block(feature)
            feat_shape = lateral_feature.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = lateral_feature + inner_top_down
            last_inner = aggr_block(last_inner)
            results.insert(0, context_block(last_inner))

        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out

class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=2):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)
        self.output_act = nn.LogSoftmax(dim=-1)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1)
        b, h, w, c = out.shape
        out = out.view(b, h, w, self.num_anchors, 2)
        out = self.output_act(out)
        
        return out.contiguous().view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=2):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1)

        return out.contiguous().view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=2):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1)

        return out.contiguous().view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, cfg, return_layers, anchor_nums=3, pretrained=True):
        super(RetinaFace, self).__init__()
        assert len(return_layers)>0,'There must be at least one return layers'

        if cfg['name'] == 'mobilenet0.25':
            backbone = mobilenet_v3_large()
            if pretrained:
                checkpoint = torch.load("./model_data_move/mobilenet_v3_large.pth",
                                        map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            backbone = resnet.__dict__[cfg['name']](pretrained=pretrained)
        assert backbone, 'Backbone can not be none!'
        self.body = _utils.IntermediateLayerGetter(backbone, return_layers)
        in_channels_stage2 = 40
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 24,
        ]
        out_channels = 40
        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels)
        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.anchors = Anchors()
        self.regressBoxes = RegressionTransform()        
        self.losslayer = losses.LossLayer()

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self,inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
        out = self.body(img_batch)
        features = self.fpn(out)
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features.values())], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features.values())], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features.values())],dim=1)
        anchors = self.anchors(img_batch)
        if self.training:
            return self.losslayer(classifications, bbox_regressions,ldm_regressions, anchors, annotations)
        else:
            bboxes, landmarks = self.regressBoxes(anchors, bbox_regressions, ldm_regressions, img_batch)
            return classifications, bboxes, landmarks

def create_retinaface(return_layers,backbone_name='mobilenet0.25',anchors_num=3,pretrained=False):

    model = RetinaFace(cfg = cfg_mnet, return_layers=return_layers, anchor_nums=3, pretrained=pretrained)

    return model



    




        






        

