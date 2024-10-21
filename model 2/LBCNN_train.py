from model import *


class BPVNet(nn.Module):
    def __init__(self, num_class):
        super(BPVNet, self).__init__()
        self.f_Net = fNet()
        self.Net1 = OneNet()
        self.Net2 = OneNet()
        self.Net3 = OneNet()
        self.Net4 = OneNet()
        self.Net5 = OneNet()

        self.fc_1 = nn.Linear(512, 512)
        self.fc_2 = nn.Linear(512, 512)
        self.fc_3 = nn.Linear(512, 512)
        self.fc_4 = nn.Linear(512, 512)
        self.fc_5 = nn.Linear(512, 512)

        self.fc1 = nn.Linear(512, num_class)
        self.fc2 = nn.Linear(512, num_class)
        self.fc3 = nn.Linear(512, num_class)
        self.fc4 = nn.Linear(512, num_class)
        self.fc5 = nn.Linear(512, num_class)
        self.fc_f = nn.Sequential(
            nn.Dropout(p=0),  # p：随机失活的比例
            # nn.Linear(512*5, 1024),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.3),
            nn.Linear(512*5, num_class))


    def forward(self, x1, x2, x3, x4, x5):
        x = self.f_Net(x1, x2, x3, x4, x5)
        x1 = self.Net1(x1)
        x2 = self.Net2(x2)
        x3 = self.Net3(x3)
        x4 = self.Net4(x4)
        x5 = self.Net5(x5)

        B, _, _, _ = x.shape
        x1 = x1.reshape(B, -1)
        x2 = x2.reshape(B, -1)
        x3 = x3.reshape(B, -1)
        x4 = x4.reshape(B, -1)
        x5 = x5.reshape(B, -1)
        x_f = x.reshape(B, -1)

        x_f_1 = self.fc_1(x_f)
        x_f_2 = self.fc_2(x_f)
        x_f_3 = self.fc_3(x_f)
        x_f_4 = self.fc_4(x_f)
        x_f_5 = self.fc_5(x_f)
        x_ff = torch.cat((x_f_1, x_f_2,x_f_3,x_f_4,x_f_5), dim=1)


        x_1 = self.fc1(x1)
        x_2 = self.fc2(x2)
        x_3 = self.fc3(x3)
        x_4 = self.fc4(x4)
        x_5 = self.fc5(x5)

        x_f = self.fc_f(x_ff)

        return x_f, x_1, x_2, x_3, x_4, x_5, x1, x2, x3, x4, x5,x_f_1,x_f_2,x_f_3,x_f_4,x_f_5
