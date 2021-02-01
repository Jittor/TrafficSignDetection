import jittor as jt
from jittor import nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    conv=nn.Conv(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
    jt.init.relu_invariant_gauss_(conv.weight, mode="fan_out")
    return conv

def conv1x1(in_planes, out_planes, stride=1):
    conv=nn.Conv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    jt.init.relu_invariant_gauss_(conv.weight, mode="fan_out")
    return conv

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if (norm_layer is None):
            norm_layer = nn.BatchNorm
        width = (int((planes * (base_width / 64.0))) * groups)
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, (planes * self.expansion))
        self.bn3 = norm_layer((planes * self.expansion))
        self.relu = nn.Relu()
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if (self.downsample is not None):
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if (norm_layer is None):
            norm_layer = nn.BatchNorm
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.feat_stride = 16
        self.out_channels = 1024

        if (replace_stride_with_dilation is None):
            replace_stride_with_dilation = [False, False, False]
        if (len(replace_stride_with_dilation) != 3):
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        stride = 2
        self.conv1 = nn.Conv(3, self.inplanes, kernel_size=7, stride=stride, padding=3, bias=False)
        jt.init.relu_invariant_gauss_(self.conv1.weight, mode="fan_out")
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.Relu()
        
        self.maxpool = nn.Pool(kernel_size=3, stride=2, padding=1, op='maximum')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=stride, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride, dilate=replace_stride_with_dilation[1])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if ((stride != 1) or (self.inplanes != (planes * block.expansion))):
            downsample = nn.Sequential(conv1x1(self.inplanes, (planes * block.expansion), stride), norm_layer((planes * block.expansion)))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = (planes * block.expansion)
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def execute(self, x):
        return self._forward_impl(x)

def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def Resnet50(pretrained=False, **kwargs):
    model = _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained: model.load("jittorhub://resnet50.pkl")
    return model

def Resnet101(pretrained=False, **kwargs):
    model =  _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained: model.load("jittorhub://resnet101.pkl")
    return model