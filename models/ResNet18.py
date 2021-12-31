import torchvision.models as models
        
class ResNet18(nn.Module):
    def __init__(self, num_classes, criterion=None):
        super(ResNet18, self).__init__()

        # Implement me
        self.num_classes = num_classes
        self.criterion = criterion
        self.net = models.resnet18(pretrained = True)
        self.conv1 = nn.Conv2d(576, 256, 5, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 64, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, num_classes, 1, 1)
        
    def forward(self, inp, gts=None):
        origShape = inp.shape

        net = self.net
        inp = net.conv1(inp)
        inp = net.bn1(inp)
        inp = net.relu(inp)
        inp = net.maxpool(inp)
        low = inp.clone()
        lshape = low.shape
        inp = net.layer1(inp)
        inp = net.layer2(inp)
        inp = net.layer3(inp)
        inp = net.layer4(inp)

        inp = F.interpolate(inp, size = (lshape[2], lshape[3]), mode = 'bilinear', 
                            align_corners = True)
        inp = torch.cat((inp, low), 1)
        
        inp = F.relu((self.conv1(inp)))
        inp = self.bn1(inp)
        inp = F.relu(self.conv2(inp))
        inp = self.bn2(inp)
        inp = F.relu(self.conv3(inp))
        inp = F.interpolate(inp, size = (origShape[2], origShape[3]), mode = 'bilinear', align_corners = True)
        # Implement me
        lfinal = inp
        if self.training:
            # Return the loss if in training mode
            return self.criterion(lfinal, gts)              
        else:
            # Return the actual prediction otherwise
            return lfinal