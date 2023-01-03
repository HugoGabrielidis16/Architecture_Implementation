import torch
import torch.nn as nn
from torchsummary import summary
from config import architecture
from torchview import draw_graph


"""
Implementation of ResUNet-a d6 model presented in the paper : 
https://www.sciencedirect.com/science/article/pii/S0924271620300149
"""


class ResBlock_a(nn.Module):
    """
    ResBlock_a implemented as seen in the paper, consist of len(d) parrallel block of each using a different dilation rate for
    the convulution.

    Args
    ----
    x : batched tensor of shape (n_batch, channels, width, height)

    Return
    -----
    y : Sum of the results of the different sequence, shape shape as x
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        d: list,
        kernel_size=3,
        stride=1,
        padding="same",
    ):
        super(ResBlock_a, self).__init__()

        self.block = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm2d(input_channels),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=output_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilatation,
                    ),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=output_channels,
                        out_channels=output_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilatation,
                    ),
                )
                for dilatation in d
            ]
        )

    def forward(self, x):
        result = [x]
        for block in self.block:
            rate = block(x)
            result.append(rate)
        return torch.stack(result, dim=0).sum(
            dim=0
        )  # this return the sum of all the differents results


class Combine(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=1,
        stride=1,
        padding="same",
        dilation=1,
        last=False,
    ) -> None:
        super(Combine, self).__init__()
        self.act = nn.ReLU()

        """
        Usually stacking the two would result in 3 times the number of channel since x_channel = 2* u_channel
        But this is not the case when we have the u = the result of the first conv, in this case x_channel = x_firstconv_channel
        so we have to adjust. It wouldnt be needed with keras as we only have to precise the output number of channel

        Exemple concatenating x_bottleneck : (n_batch, 1024,_,_) and u5 : (n_batch, 512, _, _) is gonna make a 
        tensor of shape (n_batch,1536, _, _) 
        """

        if not last:
            self.conv2DN = nn.Sequential(
                nn.Conv2d(
                    in_channels=3 * in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv2DN = nn.Sequential(
                nn.Conv2d(
                    in_channels=2 * in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x1, x2):
        x1 = self.act(x1)
        concat = torch.cat([x1, x2], dim=1)
        out = self.conv2DN(concat)
        return out


class ResUNet_a_down(nn.Module):

    """
    Down part of the model, from the first conv to the last down restnetblock.

    Args
    ------
    x : batched tensor of shape : (n_batch, channel,o_width,o_height)
    o_width = orignal width : Cannot be under 256
    o_height = original height = o_width

    Return
    ------
    x : result of the 1st convolution - shape : (n_batch,  32, o_width, o_height )
    u1 : result of the 1st resnetblock - shape : (n_batch, 32)
    u2 : result of the 2nd resnetblock - shape : (n_batch, 64)
    u3 : result of the 3rd resnetblock - shape : (n_batch, 128)
    u4 : result of the 4th resnetblock - shape : (n_batch, 256)
    u5 : result of the 5th resnetblock - shape : (n_batch, 512)
    u6 : result of the 6th resnetblock - shape : (n_batch, 1024)
    """

    def __init__(self, in_channels):
        super(ResUNet_a_down, self).__init__()

        self.convd_1 = nn.Conv2d(
            in_channels=in_channels, **architecture["down_conv"]["conv_1"]
        )
        self.convd_2 = nn.Conv2d(**architecture["down_conv"]["conv_2"])
        self.convd_3 = nn.Conv2d(**architecture["down_conv"]["conv_3"])
        self.convd_4 = nn.Conv2d(**architecture["down_conv"]["conv_4"])
        self.convd_5 = nn.Conv2d(**architecture["down_conv"]["conv_5"])
        self.convd_6 = nn.Conv2d(**architecture["down_conv"]["conv_6"])

        self.resblockd_1 = ResBlock_a(**architecture["down_block"]["resblock_1"])
        self.resblockd_2 = ResBlock_a(**architecture["down_block"]["resblock_2"])
        self.resblockd_3 = ResBlock_a(**architecture["down_block"]["resblock_3"])
        self.resblockd_4 = ResBlock_a(**architecture["down_block"]["resblock_4"])
        self.resblockd_5 = ResBlock_a(**architecture["down_block"]["resblock_5"])
        self.resblockd_6 = ResBlock_a(**architecture["down_block"]["resblock_6"])

    def forward(self, x):
        # Return the results of the 6 differents ResBlock and the result of the first convolution
        x = self.convd_1(x)  # Layer 1
        u1 = self.resblockd_1(x)  # Layer 2
        u2 = self.convd_2(u1)  # Layer 3
        u2 = self.resblockd_2(u2)  # Layer 4
        u3 = self.convd_3(u2)  # Layer 5
        u3 = self.resblockd_3(u3)  # Layer 6
        u4 = self.convd_4(u3)  # Layer 7
        u4 = self.resblockd_4(u4)  # Layer 8
        u5 = self.convd_5(u4)  # Layer 9
        u5 = self.resblockd_5(u5)  # Layer 10
        u6 = self.convd_6(u5)  # Layer 11
        u6 = self.resblockd_6(u6)  # Layer 12

        return x, u1, u2, u3, u4, u5, u6


class ResUnet_PSPPooling(nn.Module):
    def __init__(
        self,
        input_channels,
        factor,
        kernel_size=1,
        stride=1,
        padding="same",
    ) -> None:
        super(ResUnet_PSPPooling, self).__init__()

        """
        Bottleneck layer, we a're using the ResUNet-a d6 architecture.
        To do the 'RESTORE DIM', Upsample was used as, however when the o_width % x != 0, we do not comeback to the original width
        after the UpSample, so we had to adjust it manually.
        
        # Need to change that later 

        Args 
        -----
        x (u6): Result of the down part of the model - shape (n_batch, 1024, _, _)

        Return
        -----
        x_bottleneck : result of the block - shape (n_batch, 1024, _, _)
        """
        self.parrallel_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=x, stride=x),
                    nn.Upsample(
                        scale_factor=x,
                    ),
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=input_channels // 4,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(num_features=input_channels // 4),
                )
                for x in factor
            ]
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * input_channels,
                out_channels=input_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=input_channels),
        )

    def forward(self, x):
        out = [x]
        # out = []
        for sequence in self.parrallel_list:
            out.append(sequence(x))
        for i in range(len(out)):
            if out[i].shape[-1] != out[0].shape[-1]:
                out[i] = nn.Upsample(size=out[0].shape[-1])(
                    out[i]
                )  # because (we get a shape mismatch in the concat if the width % x != 0)
        output = torch.concat(out, axis=1)
        return self.conv(output)


class ResUNet_a_UP(nn.Module):
    def __init__(self, out_channels) -> None:
        super(ResUNet_a_UP, self).__init__()

        self.resblockup_1 = ResBlock_a(**architecture["up_block"]["resblock_1"])
        self.resblockup_2 = ResBlock_a(**architecture["up_block"]["resblock_2"])
        self.resblockup_3 = ResBlock_a(**architecture["up_block"]["resblock_3"])
        self.resblockup_4 = ResBlock_a(**architecture["up_block"]["resblock_4"])
        self.resblockup_5 = ResBlock_a(**architecture["up_block"]["resblock_5"])

        self.combine_1 = Combine(**architecture["combine"]["combine_1"])
        self.combine_2 = Combine(**architecture["combine"]["combine_2"])
        self.combine_3 = Combine(**architecture["combine"]["combine_3"])
        self.combine_4 = Combine(**architecture["combine"]["combine_4"])
        self.combine_5 = Combine(**architecture["combine"]["combine_5"])
        self.combine_6 = Combine(**architecture["combine"]["combine_6"], last=True)

        # self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample = nn.Upsample(scale_factor=2)

        self.psp = ResUnet_PSPPooling(**architecture["up_block"]["PSPpooling"])

        self.last_conv = nn.Conv2d(
            **architecture["last_conv"], out_channels=out_channels
        )
        self.soft = nn.Softmax(dim=1)

    def forward(self, x_bottleneck, x_firstconv, u1, u2, u3, u4, u5):

        x = self.upsample(x_bottleneck)  # Layer 14
        x = self.combine_1(x, u5)  # Layer 15 : Combine Layer 14 & Layer 10
        x = self.resblockup_1(x)  # Layer 16

        x = self.upsample(x)  # Layer 17
        x = self.combine_2(x, u4)  # Layer 18 : Combine Layer 16 & Layer 8
        x = self.resblockup_2(x)  # Layer 19

        x = self.upsample(x)  # Layer 20
        x = self.combine_3(x, u3)  # Layer 21 : Combine Layer 20 & Layer 6
        x = self.resblockup_3(x)  # Layer 22

        x = self.upsample(x)  # Layer 23
        x = self.combine_4(x, u2)  # Layer 24 : Combine Layer 23 % Layer 4
        x = self.resblockup_4(x)  # Layer 25

        x = self.upsample(x)  # Layer 26
        x = self.combine_5(x, u1)  # Layer 27 : Combine Layer 26 & Layer 2
        x = self.resblockup_5(x)  # Layer 28

        x = self.combine_6(x, x_firstconv)  # Layer 29 : Combine Layer 28 & Layer 1

        x = self.psp(x)  # Layer 30

        out = self.soft(self.last_conv(x))  # Layer 31

        return out


class ResUNET_a_d6(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(ResUNET_a_d6, self).__init__()

        self.encoder = ResUNet_a_down(in_channels)
        self.bottleneck = ResUnet_PSPPooling(**architecture["bottleneck"]["PSPpooling"])
        self.decoder = ResUNet_a_UP(out_channels)

    def forward(self, x):
        x_firstconv, u1, u2, u3, u4, u5, u6 = self.encoder(x)
        x_bottleneck = self.bottleneck(u6)
        output = self.decoder(x_bottleneck, x_firstconv, u1, u2, u3, u4, u5)
        return output


if __name__ == "__main__":
    print("Transform model to ONXX...")
    model = ResUNET_a_d6(3, 13)
    model.eval()
    dummy_input = torch.randn(32, 3, 256, 256)
    input_names = ["actual_input"]
    output_names = ["output"]
    torch.onnx.export(
        model,
        dummy_input,
        "ResUnet_a_d6.onnx",
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
    )
    print("Model transformed to ONNX !")
