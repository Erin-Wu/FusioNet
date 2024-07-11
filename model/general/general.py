import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import random
import numpy as np
from model.general.backbone import resnet_lz as resnet
from .losses import NormalizedFocalLossSigmoid


########################################[ Global Function ]########################################

def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, SynchronizedBatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, SynchronizedBatchNorm2d):
            m.eval()
        elif isinstance(m, nn.BatchNorm2d):
            m.eval()


def gene_map_gauss(map_dist_src, sigma=10):
    return torch.exp(-2.772588722 * (map_dist_src ** 2) / (sigma ** 2))


def gene_map_dist(map_dist_src, max_dist=255):
    return 1.0 - map_dist_src / max_dist


def my_resize(input, ref):
    return F.interpolate(input, size=(ref if isinstance(ref, (list, tuple)) else ref.size()[2:]), mode='bilinear',
                         align_corners=True)


def make_list(input):
    return [] if input is None else (list(input) if isinstance(input, (list, tuple)) else [input])


def point_resize(mask, ref, if_return_mask=True, tsh=0.5):
    mask_h, mask_w = mask.shape[2:4]
    ref_h, ref_w = ref if isinstance(ref, (tuple, list)) else ref.shape[2:4]
    indices = torch.nonzero(mask > tsh)
    if mask_h == ref_h and mask_w == ref_w:
        mask_new = mask
    else:
        mask_new = torch.zeros([mask.shape[0], mask.shape[1], ref_h, ref_w])
        if len(indices) != 0:
            indices = indices.float()
            indices[:, 2] = torch.floor(indices[:, 2] * ref_h / mask_h)
            indices[:, 3] = torch.floor(indices[:, 3] * ref_w / mask_w)
            indices = indices.long()
            mask_new[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]] = 1
    return mask_new.float().cuda() if if_return_mask else indices


def to_cuda(input):
    if isinstance(input, list):
        return [tmp.cuda() for tmp in input]
    elif isinstance(input, dict):
        return {key: input[key].cuda() for key in input}
    else:
        raise ValueError('Wrong Type!')


def get_key_sample(sample_batched, keys=['img'], if_list=True, if_cuda=True):  # 用于从输入的sample_batched中获取指定的键对应的样本数据
    if not isinstance(keys, list): keys = [keys]
    key_sample = [sample_batched[key] for key in keys] if if_list else {key: sample_batched[key] for key in keys}
    if if_cuda: key_sample = to_cuda(key_sample)
    return key_sample


def set_default_dict(src_dict, default_map):
    for k, v in default_map.items():
        if k not in src_dict:
            src_dict[k] = v


def get_click_map(sample_batched, mode='dist', click_keys=None):
    mode_split = mode.split('-')
    if len(mode_split) == 1:
        if_with_para = False
    else:
        if_with_para = True
        para = mode_split[-1]

    if mode.startswith('dist'):
        if click_keys is None: click_keys = ['pos_map_dist_src', 'neg_map_dist_src']
        click_map = torch.cat([(gene_map_dist(t, int(para)) if if_with_para else gene_map_dist(t)) for t in
                               get_key_sample(sample_batched, click_keys, if_list=True)], dim=1)
    elif mode.startswith('gauss'):
        if click_keys is None: click_keys = ['pos_map_dist_src', 'neg_map_dist_src']
        click_map = torch.cat([(gene_map_gauss(t, int(para)) if if_with_para else gene_map_gauss(t)) for t in
                               get_key_sample(sample_batched, click_keys, if_list=True)], dim=1)
    elif mode == 'point':
        if click_keys is None: click_keys = ['pos_points_mask', 'neg_points_mask']
        click_map = torch.cat(get_key_sample(sample_batched, click_keys, if_list=True), dim=1)
    elif mode == 'first_point':
        if click_keys is None: click_keys = ['first_point_mask']
        click_map = get_key_sample(sample_batched, click_keys, if_list=True)[0]
    elif mode == 'first_dist':
        if click_keys is None: click_keys = ['first_map_dist_src']
        click_map = gene_map_dist(get_key_sample(sample_batched, click_keys, if_list=True)[0])
    return click_map


def get_input(sample_batched, mode='dist'):
    img = get_key_sample(sample_batched, ['img'], if_list=True)[0]
    if mode in ['dist', 'gauss', 'point']:
        x = torch.cat((img, get_click_map(sample_batched, mode)), dim=1)
    elif mode in ['none', 'img']:
        x = img
    else:
        raise ValueError('mode')
    return x


def get_stack_flip_feat(x, if_horizontal=True):
    x_flip = torch.flip(x, [3 if if_horizontal else 2])
    return torch.cat([x, x_flip], dim=0)


def merge_stack_flip_result(x, if_horizontal=True, batch_num=1):
    return (x[:batch_num] + torch.flip(x[batch_num:2 * batch_num], [3 if if_horizontal else 2])) / 2.0


########################################[ MultiConv ]########################################

class MultiConv(nn.Module):
    def __init__(self, in_ch, channels, kernel_sizes=None, strides=None, dilations=None, paddings=None,
                 BatchNorm=nn.BatchNorm2d, if_w_bn=True, if_end_wo_relu=False, block_kind='conv'):
        super(MultiConv, self).__init__()
        self.num = len(channels)
        if kernel_sizes is None: kernel_sizes = [3 for c in channels]
        if strides is None: strides = [1 for c in channels]
        if dilations is None: dilations = [1 for c in channels]
        if paddings is None: paddings = [
            ((kernel_sizes[i] // 2) if dilations[i] == 1 else (kernel_sizes[i] // 2 * dilations[i])) for i in
            range(self.num)]
        convs_tmp = []
        for i in range(self.num):
            if block_kind == 'conv':
                if channels[i] == 1 or if_end_wo_relu:
                    convs_tmp.append(
                        nn.Conv2d(in_ch if i == 0 else channels[i - 1], channels[i], kernel_size=kernel_sizes[i],
                                  stride=strides[i], padding=paddings[i], dilation=dilations[i]))
                else:
                    if if_w_bn:
                        convs_tmp.append(nn.Sequential(
                            nn.Conv2d(in_ch if i == 0 else channels[i - 1], channels[i], kernel_size=kernel_sizes[i],
                                      stride=strides[i], padding=paddings[i], dilation=dilations[i], bias=False),
                            BatchNorm(channels[i]), nn.ReLU(inplace=True)))
                    else:
                        convs_tmp.append(nn.Sequential(
                            nn.Conv2d(in_ch if i == 0 else channels[i - 1], channels[i], kernel_size=kernel_sizes[i],
                                      stride=strides[i], padding=paddings[i], dilation=dilations[i], bias=False),
                            nn.ReLU(inplace=True)))
            elif block_kind == 'bottleneck':
                in_ch_cur = in_ch if i == 0 else channels[i - 1]
                out_ch_cur = channels[i]
                down_sample_cur = None if in_ch_cur == out_ch_cur else nn.Sequential(
                    nn.Conv2d(in_ch_cur, out_ch_cur, kernel_size=1, stride=strides[i], bias=False),
                    BatchNorm(out_ch_cur))
                convs_tmp.append(
                    resnet.Bottleneck(in_ch_cur, out_ch_cur // 4, strides[i], dilations[i], down_sample_cur, BatchNorm))

        self.convs = nn.Sequential(*convs_tmp)
        init_weight(self)

    def forward(self, x):
        return self.convs(x)


########################################[ SE ]########################################

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


########################################[ CA ]########################################
class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        def crop_tensor(tensor, target_size):
            current_size = tensor.size()
            if current_size == target_size:
                return tensor
            else:
                assert len(current_size) == len(target_size)
                assert all(cs >= ts for cs, ts in zip(current_size, target_size))
                slices = [slice(0, ts) for ts in target_size]
                return tensor[slices]

        # 全局平均池化操作
        x_global_pool_h = torch.mean(x, dim=(2, 3), keepdim=True)       # 以下参数都是在验证的时候
        x_global_pool_w = torch.mean(x, dim=(2, 3), keepdim=True)

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)        # ([1，256，1，24])
        x_w = torch.mean(x, dim=2, keepdim=True)        # ([1，256，1，34])

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
        x_cat_conv_relu_1 = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w, x_global_pool_h, x_global_pool_w), 3))))

        target_size = x_cat_conv_relu.size()
        x_cat_conv_relu = crop_tensor(x_cat_conv_relu_1, target_size)       # ([1，16，1，58])

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)       # ([1，16，1，24])    ([1，16，1，34])

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))      # ([1， 256，24，1])
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))          # ([1，256，1，34])

        channel_att = x * s_h.expand_as(x) * s_w.expand_as(x)  # 通道注意力          # ([1，256，24，34])

        spatial_att = torch.mean(channel_att, dim=1, keepdim=True)      # ([1，1，24，34])

        out = x + channel_att + spatial_att  # 引入残差连接           # ([1，256，24，34])

        return out


########################################[ CBAM ]########################################
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)           # 8,1,96,96
        max_out, _ = torch.max(x, dim=1, keepdim=True)       # 8,1,96,96
        x = torch.cat([avg_out, max_out], dim=1)        # 8,2,96,96
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


########################################[ ECA ]########################################

class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


########################################[ MyASPP ]########################################

class _ASPPModule(nn.Module):
    def __init__(self, module, inplanes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(_ASPPModule, self).__init__()

        self.module = module
        if self.module == 1:
            self.atrous_conv = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0,
                                     dilation=1, bias=False))
        elif self.module == 2:
            self.atrous_conv = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1,
                                                       dilation=1, bias=False))
        elif self.module == 3:
            self.atrous_conv = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=2,
                                                       dilation=2, bias=False),
                                             nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=2,
                                                       dilation=2, bias=False))
        elif self.module == 4:
            self.atrous_conv = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=3,
                                                       dilation=3, bias=False),
                                             nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=3,
                                                       dilation=3, bias=False),
                                             nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=3,
                                                       dilation=3, bias=False))
        else:
            self.atrous_conv = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=5,
                                                       dilation=5, bias=False),
                                             nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=5,
                                                       dilation=5, bias=False),
                                             nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=5,
                                                       dilation=5, bias=False),
                                             nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=5,
                                                       dilation=5, bias=False))


        self.bn = BatchNorm(outplanes)
        self.relu = nn.ReLU(inplace=True)
        init_weight(self)

    def forward(self, x):
        x = self.atrous_conv(x)
        x= self.bn(x)
        x = self.relu(x)
        return x


class MyASPP(nn.Module):
    def __init__(self, in_ch, out_ch, dilations, BatchNorm=nn.BatchNorm2d, if_global=True):
        super(MyASPP, self).__init__()
        self.if_global = if_global

        if len(dilations) == 4 and dilations[0] == 1:
            dilations = dilations[1:]
        assert len(dilations) == 3

        self.aspp1 = _ASPPModule(1, in_ch, out_ch)
        self.aspp2 = _ASPPModule(2, in_ch, out_ch)
        self.aspp3 = _ASPPModule(3, in_ch, out_ch)
        self.aspp4 = _ASPPModule(4, in_ch, out_ch)
        self.aspp5 = _ASPPModule(5, in_ch, out_ch)

        if if_global:
            self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                 nn.Conv2d(in_ch, out_ch, 1, stride=1, bias=False),
                                                 BatchNorm(out_ch),
                                                 nn.ReLU(inplace=True))
        self.ca = CA_Block(out_ch)

        merge_channel = out_ch * 6 if if_global else out_ch * 4

        self.conv1 = nn.Conv2d(merge_channel, out_ch, 1, bias=False)
        self.bn1 = BatchNorm(out_ch)
        self.relu = nn.ReLU(inplace=True)
        init_weight(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.ca(x1)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.aspp5(x)

        if self.if_global:
            x6 = self.global_avg_pool(x)
            x6 = F.interpolate(x6, size=x4.size()[2:], mode='bilinear', align_corners=True)
            x6 = self.ca(x6)
            x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        else:
            x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


########################################[ MyDecoder ]########################################

class MyDecoder(nn.Module):
    def __init__(self, in_ch, in_ch_reduce, side_ch, side_ch_reduce, out_ch, BatchNorm=nn.BatchNorm2d, size_ref='side'):
        super(MyDecoder, self).__init__()
        self.size_ref = size_ref
        self.relu = nn.ReLU(inplace=True)
        self.in_ch_reduce, self.side_ch_reduce = in_ch_reduce, side_ch_reduce

        if in_ch_reduce is not None:
            self.in_conv = nn.Sequential(nn.Conv2d(in_ch, in_ch_reduce, 1, bias=False), BatchNorm(in_ch_reduce),
                                         nn.ReLU(inplace=True))
        if side_ch_reduce is not None:
            self.side_conv = nn.Sequential(nn.Conv2d(side_ch, side_ch_reduce, 1, bias=False), BatchNorm(side_ch_reduce),
                                           nn.ReLU(inplace=True))

        # 改动部分
        self.eca = eca_block(channel=in_ch)
        self.sa = SpatialAttention(kernel_size=7)

        merge_ch = (in_ch_reduce if in_ch_reduce is not None else in_ch) + (
            side_ch_reduce if side_ch_reduce is not None else side_ch)

        self.merge_conv1 = nn.Sequential(nn.Conv2d(merge_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                                         BatchNorm(out_ch),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                                         BatchNorm(out_ch),
                                         nn.ReLU(inplace=True))

        self.merge_conv2 = nn.Sequential(nn.Conv2d(256, out_ch, kernel_size=3, stride=1, padding=1, bias=False),        # 这里的256通道是看了下上一个的输出
                                       BatchNorm(out_ch),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(out_ch),
                                       nn.ReLU(inplace=True))
        self.conv = nn.Conv2d(in_channels=256,out_channels=304,kernel_size=1)
        init_weight(self)

    def forward(self, input, side):
        if self.in_ch_reduce is not None:
            input = self.in_conv(input)
        if self.side_ch_reduce is not None:
            side = self.side_conv(side)

        if self.size_ref == 'side':
            input = F.interpolate(input, size=side.size()[2:], mode='bilinear', align_corners=True)
        elif self.size_ref == 'input':
            side = F.interpolate(side, size=input.size()[2:], mode='bilinear', align_corners=True)

        # 改动部分
        x1 = self.eca(input)
        x2 = self.sa(side)

        merge = torch.cat((input, side), dim=1)

        # 改动部分
        merge = torch.mul(merge, x2)
        x1 = self.conv(x1)
        merge = torch.mul(merge, x1)
        output = self.merge_conv1(merge)
        return output


########################################[ PredDecoder ]########################################

class PredDecoder(nn.Module):
    def __init__(self, in_ch, layer_num=1, BatchNorm=nn.BatchNorm2d, if_sigmoid=False):
        super(PredDecoder, self).__init__()
        convs_tmp = []
        for i in range(layer_num - 1):
            convs_tmp.append(nn.Sequential(nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                           BatchNorm(in_ch // 2), nn.ReLU(inplace=True)))
            in_ch = in_ch // 2
        convs_tmp.append(nn.Conv2d(in_ch, 1, kernel_size=1, stride=1))
        if if_sigmoid: convs_tmp.append(nn.Sigmoid())
        self.pred_conv = nn.Sequential(*convs_tmp)
        init_weight(self)

    def forward(self, input):
        return self.pred_conv(input)


########################################[ Template for IIS ]########################################

class MyNetBase(nn.Module):
    def __init__(self, special_lr=0.1, remain_lr=1.0, size=512, aux_parameter={}):
        super(MyNetBase, self).__init__()
        self.diy_lr = []
        self.special_lr = special_lr
        self.remain_lr = remain_lr
        self.size = size
        self.aux_parameter = aux_parameter
        self.loss = NormalizedFocalLossSigmoid(gamma=0.5)

    def get_params(self, modules):  # 获取指定模块中需要求梯度的参数。
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1],
                                                                                                 SynchronizedBatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_train_params_lr(self, lr):  # 根据设定的学习率和不同模块的比例，获取训练参数和学习率。
        train_params, special_modules = [], []
        for modules, ratio in self.diy_lr:
            train_params.append({'params': self.get_params(modules), 'lr': lr * ratio})
            special_modules += modules
        remain_modules = [module for module in self.children() if module not in special_modules]
        train_params.append({'params': self.get_params(remain_modules), 'lr': lr * self.remain_lr})
        return train_params

    # can change
    def get_loss(self, output, gt=None, sample_batched=None, others=None):  # 计算模型输出与目标值之间的损失值。
        if gt is None: gt = sample_batched['gt'].cuda()
        losses = [self.loss(my_resize(t, gt), gt) for t in make_list(output)]
        return losses

    def get_loss_union(self, output, gt=None, sample_batched=None, others=None):  # 计算模型输出与目标值之间的损失值，并进行反向传播。
        losses = self.get_loss(output, gt, sample_batched, others)
        loss_items = torch.Tensor([loss.item() for loss in losses]).unsqueeze(0)
        losses = sum(losses)
        losses.backward()
        return loss_items

    def get_result(self, output, index=None):
        result = torch.sigmoid((make_list(output))[0]).data.cpu().numpy()  # 对模型输出进行sigmoid变换，将其值域缩放到[0, 1]范围内；
        return result[:, 0, :, :] if index is None else result[index, 0, :, :]  # 如果index为None，则返回整个数组；否则，返回指定索引的部分数组。

    def forward_union(self, sample_batched, mode='train'):  # 前向传播函数，根据模式返回输出结果和损失值（如果是训练模式）。
        output = self.forward(sample_batched, mode)
        if mode == 'train':
            loss_items = self.get_loss(output, sample_batched=sample_batched)
            return output, loss_items
        else:
            return output


########################################[ Template for FocusCut ]########################################

class MyNetBaseHR(MyNetBase):
    def __init__(self, input_channel=5, output_stride=16, if_sync_bn=False, if_freeze_bn=False, special_lr=0.1,
                 remain_lr=1.0, size=512, aux_parameter={}):
        super(MyNetBaseHR, self).__init__(special_lr, remain_lr, size, aux_parameter)

        BatchNorm = SynchronizedBatchNorm2d if if_sync_bn else nn.BatchNorm2d
        default_map = {'backbone': 'resnet50', 'point_map': 'gauss', 'into_layer': -1, 'pretrained': True,
                       'if_pre_pred': True, 'weight_loss': False, 'backward_each': False}
        set_default_dict(self.aux_parameter, default_map)

        side_chs = [64, 256, 512, 1024, 2048] if self.aux_parameter['backbone'] in ['resnet50', 'resnet101',
                                                                                    'resnet152'] else [64, 64, 128, 256,
                                                                                                       512]
        self.backbone = resnet.get_resnet_backbone(self.aux_parameter['backbone'], output_stride, BatchNorm,
                                                   self.aux_parameter['pretrained'], (
                                                       3 if self.aux_parameter['into_layer'] >= 0 else (
                                                                   5 + int(self.aux_parameter['if_pre_pred']))), True)
        self.my_aspp = MyASPP(in_ch=side_chs[-1], out_ch=side_chs[-1] // 8,
                              dilations=[int(i * size / 512 + 0.5) * (16 // output_stride) for i in [6, 12, 18]],
                              BatchNorm=BatchNorm, if_global=True)
        self.my_decoder = MyDecoder(in_ch=side_chs[-1] // 8, in_ch_reduce=None, side_ch=side_chs[-1] // 8,
                                    side_ch_reduce=side_chs[1] // 16 * 3, out_ch=side_chs[-1] // 8, BatchNorm=BatchNorm)
        self.pred_decoder = PredDecoder(in_ch=side_chs[-1] // 8, BatchNorm=BatchNorm)
        self.conv1x1 = nn.Conv2d(512, 256, kernel_size=1)

        if self.aux_parameter['into_layer'] != -1:
            in_ch = 3 if self.aux_parameter['into_layer'] == 0 else side_chs[self.aux_parameter['into_layer'] - 1]
            self.encoder_anno = nn.Sequential(
                nn.Conv2d(in_ch + 2 + int(self.aux_parameter['if_pre_pred']), in_ch, 1, bias=False), BatchNorm(in_ch),
                nn.ReLU(inplace=True))

        self.diy_lr = [[[self.backbone], self.special_lr]]
        if if_freeze_bn: freeze_bn(self)

        self.side_chs = side_chs

    # return_list ['img',0,1,2,3,4,'aspp','decoder','pred_decoder']
    def backbone_forward(self, sample_batched, img_key, click_keys, pre_pred_key, return_list=['final']):
        def return_results_func(key, tmp):
            if key in return_list: return_results.append(tmp)
            return len(return_results) == len(return_list)

        def resize_l2(l1, l2):
            new_height = l1.size(2)
            new_width = l1.size(3)

            l2_resized = torch.nn.functional.interpolate(l2, size=(new_height, new_width), mode='bilinear', align_corners=False)

            return l2_resized

        return_results = []

        img = sample_batched[img_key].cuda()  # 根据img_key从sample_batched中获取图片数据
        if return_results_func('img', img): return return_results

        click_map = get_click_map(sample_batched, self.aux_parameter['point_map'],
                                  click_keys=click_keys)  # 根据指定的click_keys从sample_batched中获取点击图数据
        if return_results_func('click_map', click_map): return return_results

        pre_pred = sample_batched[pre_pred_key].cuda()  # 根据pre_pred_key从sample_batched中获取预测数据
        if return_results_func('pre_pred', pre_pred): return return_results

        aux = torch.cat([click_map, pre_pred], dim=1) if self.aux_parameter[
            'if_pre_pred'] else click_map  # 根据if_pre_pred的设置，将点击图和预测数据拼接在一起作为辅助输入（aux）
        if return_results_func('aux', aux): return return_results

        x = img

        if self.aux_parameter['into_layer'] == -1:  # 根据设置的into_layer值，将图像和辅助输入拼接在一起。
            x = torch.cat((x, my_resize(aux, x)), dim=1)

        for i in range(5):  # 在循环的过程中，根据不同的循环次数，将输入数据传入不同的网络层进行处理。
            if i == 1: x = self.backbone.layers[0][-1](x)

            if self.aux_parameter['into_layer'] == i:
                x = self.encoder_anno(torch.cat((x, my_resize(aux, x)), dim=1))

            x = self.backbone.layers[i][:-1](x) if i == 0 else self.backbone.layers[i](x)

            if i == 1: l1 = x

            if i in return_list:
                if return_results_func(i, x):
                    return return_results

        if self.aux_parameter['into_layer'] == 5:  # 根据into_layer的设置，将辅助输入和输出结果拼接在一起，并传入encoder_anno进行处理。
            x = self.encoder_anno(torch.cat((x, my_resize(aux, x)), dim=1))

        x = self.my_aspp(x)
        if return_results_func('aspp', x): return return_results

        x = self.my_decoder(x, l1)
        if return_results_func('decoder', x): return return_results

        x = self.pred_decoder(x)
        if return_results_func('pred_decoder', x): return return_results

        x = my_resize(x, img)  # 对输出结果进行调整大小处理，并返回。
        if return_results_func('final', x): return return_results

    def wo_forward(self, sample_batched):
        return self.backbone_forward(sample_batched, 'img', ['pos_map_dist_src', 'neg_map_dist_src'], 'pre_pred')[0]

    def hr_forward(self, sample_batched):
        return \
        self.backbone_forward(sample_batched, 'img_hr', ['pos_map_dist_src_hr', 'neg_map_dist_src_hr'], 'pre_pred_hr')[
            0]

    def forward(self, sample_batched, mode='train'):
        if mode == 'eval': mode = 'eval-wo'
        results, losses = [], []

        for part in ['wo', 'hr']:
            if mode in ['train', 'eval-{}'.format(part)]:
                result_part = getattr(self, '{}_forward'.format(part))(sample_batched)
                if result_part is not None:
                    results.append(result_part[0] if isinstance(result_part, (list, tuple)) else result_part)
                    if mode in ['train']:
                        loss_part = getattr(self, 'get_{}_loss'.format(part))(result_part, sample_batched)
                        losses.append(loss_part)
                        if self.aux_parameter['backward_each']:
                            loss_part.backward()

        if mode in ['train']:
            loss_items = torch.Tensor([loss.item() for loss in losses]).unsqueeze(0).cuda()
            if not self.aux_parameter['backward_each']:
                losses_sum = sum(losses)
                losses_sum.backward()
            return results, loss_items
        else:
            return results

    def get_wo_loss(self, output_wo, sample_batched):
        gt = sample_batched['gt'].cuda()
        wo_loss = F.binary_cross_entropy_with_logits(output_wo, gt)
        return wo_loss

    def get_hr_loss(self, output_hr, sample_batched):
        weight = sample_batched['gt_weight_hr'].cuda() if self.aux_parameter['weight_loss'] else None
        gt_hr = sample_batched['gt_hr'].cuda()
        hr_loss = F.binary_cross_entropy_with_logits(output_hr, gt_hr, weight=weight)
        return hr_loss

    def get_hr_params_lr(self, lr):
        train_params = []
        ignore_modules = [self.backbone, self.my_aspp, self.my_decoder, self.pred_decoder]
        if self.aux_parameter['into_layer'] != -1: ignore_modules.append(self.encoder_anno)
        remain_modules = [module for module in self.children() if module not in ignore_modules]
        train_params.append({'params': self.get_params(remain_modules), 'lr': lr})
        return train_params

    def freeze_main_bn(self):
        freeze_modules = [self.backbone, self.my_aspp, self.my_decoder, self.pred_decoder]  # 把这些参数保存到模块中
        if self.aux_parameter['into_layer'] != -1: freeze_modules.append(self.encoder_anno)
        for module in freeze_modules:
            freeze_bn(module)


