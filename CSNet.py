import torch

VIDEO_WIDTH, VIDEO_HEIGHT=704, 576
use_cuda = True
device = torch.device("cuda:0" if (
    use_cuda and torch.cuda.is_available()) else "cpu")
path = "./DIV2K_HR/"
save_name = "ckpts/ColorLR.pth"
num_epoch = 500
learning_rates = (1e-4, 2e-5)


class CSNet(torch.nn.Module):
    def __init__(self, training_out=False):
        super(CSNet, self).__init__()
        self.training_out = training_out
        # for 1920x1080, 24 can be devided; 
        # for 720p, choose 20
        # for 704p, choose 16
        # for 288p, choose 16 or 8
        k_stride = 16
        # MR = 0.01
        color_channel = 3
        mr = 12  # int(MR * 576) -- or 400; color *3
        self.conv0 = torch.nn.Conv2d(
            in_channels=color_channel, out_channels=mr, kernel_size=(2*k_stride), stride=k_stride, padding=k_stride)
        self.deconv0 = torch.nn.ConvTranspose2d(
            in_channels=mr, out_channels=color_channel, kernel_size=(2*k_stride), stride=k_stride, padding=k_stride)

        self.conv1_1 = torch.nn.Conv2d(
            in_channels=color_channel, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.conv1_2 = torch.nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv1_3 = torch.nn.Conv2d(
            in_channels=32, out_channels=color_channel, kernel_size=7, stride=1, padding=3)

        self.conv2_1 = torch.nn.Conv2d(
            in_channels=color_channel, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.conv2_2 = torch.nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_3 = torch.nn.Conv2d(
            in_channels=32, out_channels=color_channel, kernel_size=7, stride=1, padding=3)

        self.conv3_1 = torch.nn.Conv2d(
            in_channels=color_channel, out_channels=64, kernel_size=11, stride=1, padding=5)
        self.conv3_2 = torch.nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3_3 = torch.nn.Conv2d(
            in_channels=32, out_channels=color_channel, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        measurement = self.conv0(x)
        y0 = self.deconv0(measurement)
        y = torch.nn.functional.relu(self.conv1_1(y0))
        y = torch.nn.functional.relu(self.conv1_2(y))
        y1 = y0 + self.conv1_3(y)

        y = torch.nn.functional.relu(self.conv2_1(y1))
        y = torch.nn.functional.relu(self.conv2_2(y))
        y2 = y1 + self.conv2_3(y)

        y = torch.nn.functional.relu(self.conv3_1(y2))
        y = torch.nn.functional.relu(self.conv3_2(y))
        y = y2 + self.conv3_3(y)
        if self.training_out:
            return y0, y1, y2, y
        else:
            return measurement, y  # y0, y1, y2, y
    def compress(self,x):
        return self.conv0(x)

    def recon(self, measurement):
        y0 = self.deconv0(measurement)
        y = torch.nn.functional.relu(self.conv1_1(y0))
        y = torch.nn.functional.relu(self.conv1_2(y))
        y1 = y0 + self.conv1_3(y)

        y = torch.nn.functional.relu(self.conv2_1(y1))
        y = torch.nn.functional.relu(self.conv2_2(y))
        y2 = y1 + self.conv2_3(y)

        y = torch.nn.functional.relu(self.conv3_1(y2))
        y = torch.nn.functional.relu(self.conv3_2(y))
        y = y2 + self.conv3_3(y)


        return y  # y0, y1, y2, y