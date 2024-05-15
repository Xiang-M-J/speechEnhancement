import torch
import torch.nn as nn


class crn_net(nn.Module):
    def __init__(self, fft_size):
        super(crn_net, self).__init__()
        self.en = Encoder()
        self.freq_len = self.get_freq_len(fft_size)
        
        self.lstm = nn.LSTM(self.freq_len * 256, self.freq_len * 256, 2, batch_first=True)
        self.de = Decoder()

    def get_freq_len(self, fft_size):
        freq_len = fft_size // 2 + 1
        for i in range(5):
            freq_len = (freq_len - 3) // 2 + 1
        return freq_len
    def forward(self, x):
        x = x.unsqueeze(dim=1)
        batch_size, _, seq_len, _ = x.shape
        x, x_list = self.en(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, seq_len, -1)     # batch_size seq_len  1024
        x, _ = self.lstm(x)
        x = x.view(batch_size, seq_len, 256, self.freq_len)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = self.de(x, x_list)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        pad1 = nn.ConstantPad2d((0, 0, 1, 0), value= 0.)
        en1 = nn.Sequential(
           pad1,
           nn.Conv2d(1, 16, kernel_size=(2, 3), stride=(1, 2)),
           nn.BatchNorm2d(16),
           nn.ELU())
        en2 = nn.Sequential(
           pad1,
           nn.Conv2d(16, 32, kernel_size=(2, 3), stride=(1, 2)),
           nn.BatchNorm2d(32),
           nn.ELU())
        en3 = nn.Sequential(
           pad1,
           nn.Conv2d(32, 64, kernel_size=(2, 3), stride=(1, 2)),
           nn.BatchNorm2d(64),
           nn.ELU())
        en4 = nn.Sequential(
           pad1,
           nn.Conv2d(64, 128, kernel_size=(2, 3),stride=(1, 2)),
           nn.BatchNorm2d(128),
           nn.ELU())
        en5 = nn.Sequential(
           pad1,
           nn.Conv2d(128, 256, kernel_size=(2, 3), stride=(1, 2)),
           nn.BatchNorm2d(256),
           nn.ELU())
        self.en_module = nn.ModuleList([en1, en2, en3, en4, en5])

    def forward(self, x):
        x_list = []
        for i in range(len(self.en_module)):
            x = self.en_module[i](x)
            x_list.append(x)
        return x, x_list

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        pad1 = nn.ConstantPad2d((1, 0, 0, 0), value=0.)
        de1 = nn.Sequential(
            nn.ConvTranspose2d(256*2, 128, kernel_size=(2, 3), stride=(1, 2)),
            Chomp_T(1),
            nn.BatchNorm2d(128),
            nn.ELU())
        de2 = nn.Sequential(
            nn.ConvTranspose2d(128*2, 64, kernel_size=(2, 3), stride=(1, 2)),
            Chomp_T(1),
            nn.BatchNorm2d(64),
            nn.ELU())
        de3 = nn.Sequential(
            nn.ConvTranspose2d(64*2, 32, kernel_size=(2, 3), stride=(1, 2)),
            Chomp_T(1),
            nn.BatchNorm2d(32),
            nn.ELU())
        de4 = nn.Sequential(
            nn.ConvTranspose2d(32*2, 16, kernel_size=(2, 3), stride=(1, 2)),
            pad1,
            Chomp_T(1),
            nn.BatchNorm2d(16),
            nn.ELU())
        de5 = nn.Sequential(
            nn.ConvTranspose2d(16*2, 1, kernel_size=(2, 3), stride=(1, 2)),
            Chomp_T(1),
            nn.BatchNorm2d(1),
            nn.Softplus())
        self.de_module = nn.ModuleList([de1, de2, de3, de4, de5])

    def forward(self, x, x_list):
        for i in range(len(self.de_module)):
            x = torch.cat((x, x_list[-(i+1)]), dim=1)
            x = self.de_module[i](x)
        return x.squeeze(dim=1)


class Chomp_T(nn.Module):
    def __init__(self, chomp_t):
        super(Chomp_T, self).__init__()
        self.chomp_t = chomp_t
    def forward(self, x):
        return x[:, :, 0:-self.chomp_t, :]

if __name__ == "__main__":
    x = torch.rand([4, 243, 161])
    model = crn_net()
    y = model(x)
    print(y.shape)