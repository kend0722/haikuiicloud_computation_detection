from torch import nn,softmax
from torchvision.models import mobilenet_v2

class MobilenetV2(nn.Module):

    def __init__(self, kpt_num=2,):
        super(MobilenetV2, self).__init__()
        self.backbone = mobilenet_v2(pretrained=True)  # pretrained:预先训练的，预训练参数，等于True,说明采用预训练参数
        self.linear1 = nn.Linear(in_features=1000, out_features=512)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(0.05)
        self.linear2 = nn.Linear(in_features=512, out_features=kpt_num)

    def forward(self, x):
        x = self.backbone(x)
        # x = self.dropout(x)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = softmax(x, dim=1)  # 不需要手动加，在交叉熵损失计算时，会自动加入
        return x

    
    
