import torch
import torch.nn as nn
import torch.nn.functional as F

# ======= MLP  =======
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        return x


# ======= main =======
class twoDTSAD(nn.Module):
    def __init__(self, win_size, d_model=128,
                 local_size=[3, 5, 7], global_size=[3, 5, 7],
                 channel=55, dropout=0.05):

        super(twoDTSAD, self).__init__()
        self.local_size = local_size
        self.global_size = global_size
        self.channel = channel
        self.win_size = win_size


        self.delay_conv = nn.Conv2d(
            channel, channel,
            kernel_size=(local_size[0], 1),
            padding=0, groups=channel
        )

        self.gaf_conv = nn.Conv2d(
            channel, channel,
            kernel_size=(local_size[0], 1),
            padding=0, groups=channel
        )

        self.delay_conv_T = nn.Conv2d(
            channel, channel,
            kernel_size=(1, local_size[0]),
            padding=0, groups=channel
        )


        self.gaf_conv_T = nn.Conv2d(
            channel, channel,
            kernel_size=(1, local_size[0]),
            padding=0, groups=channel
        )


        self.mlp_local = MLP(
            local_size[0] * 4,
            d_model, 1
        )

        self.mlp_global = MLP(
            (global_size[0]-local_size[0]+1)*global_size[0] * 4,
            d_model, 1
        )

        # 下采样反卷积保持不变
        self.delay_local_decoder = nn.ConvTranspose2d(
            channel, channel,
            kernel_size=(local_size[0] - 1, 1),
            padding=0, groups=channel
        )
        self.delay_global_decoder = nn.ConvTranspose2d(
            channel, channel,
            kernel_size=(2*local_size[0]-global_size[0]-1, 1),
            padding=0, groups=channel
        )
        self.gaf_local_decoder = nn.ConvTranspose2d(
            channel, channel,
            kernel_size=(local_size[0] - 1, 1),
            padding=0, groups=channel
        )
        self.gaf_global_decoder = nn.ConvTranspose2d(
            channel, channel,
            kernel_size=(2 * local_size[0] - global_size[0] - 1, 1),
            padding=0, groups=channel
        )

    def conv_with_new_T_kernel(self, x, conv_layer, conv_layer_T):

        fe1 = conv_layer(x)
        fe2 = conv_layer_T(x)
        return fe1, fe2


    def forward(self, B, L, M, local_delay, global_delay, local_gaf, global_gaf):

        # ---- delay ----
        local_fe1, local_fe2 = self.conv_with_new_T_kernel(local_delay, self.delay_conv, self.delay_conv_T)
        global_fe1, global_fe2 = self.conv_with_new_T_kernel(global_delay, self.delay_conv, self.delay_conv_T)

        local_fe2 = local_fe2.permute(0, 1, 3, 2)
        global_fe2 = global_fe2.permute(0, 1, 3, 2)

        delay_local = torch.cat([local_fe1, local_fe2], dim=2)
        delay_global = torch.cat([global_fe1, global_fe2], dim=2)

        delay_local_deco = self.delay_local_decoder(delay_local)
        delay_global_deco = self.delay_global_decoder(delay_global)

        # ---- GAF ----
        local_gaf_fe1, local_gaf_fe2 = self.conv_with_new_T_kernel(local_gaf, self.gaf_conv, self.gaf_conv_T)
        global_gaf_fe1, global_gaf_fe2 = self.conv_with_new_T_kernel(global_gaf, self.gaf_conv, self.gaf_conv_T)

        local_gaf_fe2 = local_gaf_fe2.permute(0, 1, 3, 2)
        global_gaf_fe2 = global_gaf_fe2.permute(0, 1, 3, 2)

        gaf_local = torch.cat([local_gaf_fe1, local_gaf_fe2], dim=2)
        gaf_global = torch.cat([global_gaf_fe1, global_gaf_fe2], dim=2)

        gaf_local_deco = self.gaf_local_decoder(gaf_local)
        gaf_global_deco = self.gaf_global_decoder(gaf_global)

        # ---- reshape ----
        local_fe1 = local_fe1.reshape(B * L, M, -1)
        local_fe2 = local_fe2.reshape(B * L, M, -1)
        global_fe1 = global_fe1.reshape(B * L, M, -1)
        global_fe2 = global_fe2.reshape(B * L, M, -1)

        local_gaf_fe1 = local_gaf_fe1.reshape(B * L, M, -1)
        local_gaf_fe2 = local_gaf_fe2.reshape(B * L, M, -1)
        global_gaf_fe1 = global_gaf_fe1.reshape(B * L, M, -1)
        global_gaf_fe2 = global_gaf_fe2.reshape(B * L, M, -1)

        # ---- MLP ----
        local_final_fe = torch.cat(
            [local_fe1, local_fe2, local_gaf_fe1, local_gaf_fe2], dim=2
        )
        global_final_fe = torch.cat(
            [global_fe1, global_fe2, global_gaf_fe1, global_gaf_fe2], dim=2
        )

        local_fe = self.mlp_local(local_final_fe).squeeze(2)
        global_fe = self.mlp_global(global_final_fe).squeeze(2)

        return (local_fe.view(B, L, M),
                global_fe.view(B, L, M),
                delay_local_deco,
                delay_global_deco,
                gaf_local_deco,
                gaf_global_deco)
