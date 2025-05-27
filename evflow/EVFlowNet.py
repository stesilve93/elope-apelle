import torch
from torch import nn

_BASE_CHANNELS = 64

class build_resnet_block(nn.Module):
    """
    a resnet block which includes two general_conv2d
    """
    def __init__(self, channels, layers=2, do_batch_norm=False):
        super(build_resnet_block,self).__init__()
        self._channels = channels
        self._layers = layers

        self.res_block = nn.Sequential(*[general_conv2d(in_channels=self._channels,
                                             out_channels=self._channels,
                                             strides=1,
                                             do_batch_norm=do_batch_norm) for i in range(self._layers)])

    def forward(self,input_res):
        inputs = input_res.clone()
        input_res = self.res_block(input_res)
        return input_res + inputs

class upsample_conv2d_and_predict_flow(nn.Module):
    """
    an upsample convolution layer which includes a nearest interpolate and a general_conv2d
    """
    def __init__(self, in_channels, out_channels, ksize=3, do_batch_norm=False):
        super(upsample_conv2d_and_predict_flow, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._ksize = ksize
        self._do_batch_norm = do_batch_norm

        self.general_conv2d = general_conv2d(in_channels=self._in_channels,
                                             out_channels=self._out_channels,
                                             ksize=self._ksize,
                                             strides=1,
                                             do_batch_norm=self._do_batch_norm,
                                             padding=0)
        
        self.pad = nn.ReflectionPad2d(padding=(int((self._ksize-1)/2), int((self._ksize-1)/2),
                                        int((self._ksize-1)/2), int((self._ksize-1)/2)))#对称padding

        self.predict_flow = general_conv2d(in_channels=self._out_channels,
                                           out_channels=2,
                                           ksize=1,
                                           strides=1,
                                           padding=0,
                                           activation='tanh')

    def forward(self, conv):
        shape = conv.shape
        conv = nn.functional.interpolate(conv,size=[shape[2]*2,shape[3]*2],mode='nearest')#最近邻插值上采样
        conv = self.pad(conv)
        conv = self.general_conv2d(conv)

        flow = self.predict_flow(conv) * 256.
        
        return torch.cat([conv,flow.clone()], dim=1), flow

def general_conv2d(in_channels,out_channels, ksize=3, strides=2, padding=1, do_batch_norm=False, activation='relu'):
    """
    a general convolution layer which includes a conv2d, a relu and a batch_normalize
    """
    if activation == 'relu':
        if do_batch_norm:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels,eps=1e-5,momentum=0.99)
            )
        else:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding),
                nn.ReLU(inplace=True)
            )
    elif activation == 'tanh':
        if do_batch_norm:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding),
                nn.Tanh(),
                nn.BatchNorm2d(out_channels,eps=1e-5,momentum=0.99)
            )
        else:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding),
                nn.Tanh()
            )
    return conv2d

class EVFlowNet(nn.Module):
    def __init__(self, args):
        super(EVFlowNet,self).__init__()
        self._args = args

        self.encoder1 = general_conv2d(in_channels = 4, out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder2 = general_conv2d(in_channels = _BASE_CHANNELS, out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder3 = general_conv2d(in_channels = 2*_BASE_CHANNELS, out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder4 = general_conv2d(in_channels = 4*_BASE_CHANNELS, out_channels=8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.resnet_block = nn.Sequential(*[build_resnet_block(8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm) for i in range(2)])

        self.decoder1 = upsample_conv2d_and_predict_flow(in_channels=16*_BASE_CHANNELS,
                        out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder2 = upsample_conv2d_and_predict_flow(in_channels=8*_BASE_CHANNELS+2,
                        out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder3 = upsample_conv2d_and_predict_flow(in_channels=4*_BASE_CHANNELS+2,
                        out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder4 = upsample_conv2d_and_predict_flow(in_channels=2*_BASE_CHANNELS+2,
                        out_channels=int(_BASE_CHANNELS/2), do_batch_norm=not self._args.no_batch_norm)

    def forward(self,inputs):
        # encoder
        skip_connections = {}
        inputs = self.encoder1(inputs)
        skip_connections['skip0'] = inputs.clone()
        inputs = self.encoder2(inputs)
        skip_connections['skip1'] = inputs.clone()
        inputs = self.encoder3(inputs)
        skip_connections['skip2'] = inputs.clone()
        inputs = self.encoder4(inputs)
        skip_connections['skip3'] = inputs.clone()

        # transition
        inputs = self.resnet_block(inputs)

        # decoder
        flow_dict = {}
        inputs = torch.cat([inputs, skip_connections['skip3']], dim=1)
        inputs, flow = self.decoder1(inputs)
        flow_dict['flow0'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip2']], dim=1)
        inputs, flow = self.decoder2(inputs)
        flow_dict['flow1'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip1']], dim=1)
        inputs, flow = self.decoder3(inputs)
        flow_dict['flow2'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip0']], dim=1)
        inputs, flow = self.decoder4(inputs)
        flow_dict['flow3'] = flow.clone()

        return flow_dict
        

if __name__ == "__main__":
    from config import configs
    import time
    from data_loader import EventData
    '''
    args = configs()
    model = EVFlowNet(args).cuda()
    input_ = torch.rand(8,4,256,256).cuda()
    a = time.time()
    output = model(input_)
    b = time.time()
    print(b-a)
    print(output['flow0'].shape, output['flow1'].shape, output['flow2'].shape, output['flow3'].shape)
    #print(model.state_dict().keys())
    #print(model)
    '''
    import numpy as np
    args = configs()
    model = EVFlowNet(args).cuda()
    EventDataset = EventData(args.data_path, 'train')
    EventDataLoader = torch.utils.data.DataLoader(dataset=EventDataset, batch_size=args.batch_size, shuffle=True)
    #model = nn.DataParallel(model)
    #model.load_state_dict(torch.load(args.load_path+'/model18'))
    for input_, _, _, _ in EventDataLoader:
        input_ = input_.cuda()
        a = time.time()
        (model(input_))
        b = time.time()
        print(b-a)