import torch
import torch.nn as nn

from renderer.modules import DownConvResBlock, ResBlock, UpConvResBlock, ConvResBlock
from renderer.attention_modules import CrossAttention, SelfAttention
from renderer.lia_resblocks import StyledConv,EqualConv2d,EqualLinear

class IdentityEncoder(nn.Module):
    def __init__(self, in_channels=3, output_channels=[64, 128, 256, 512, 512, 512], initial_channels=32, dm=512):
        super(IdentityEncoder, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, initial_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )
        self.down_block_0 = DownConvResBlock(initial_channels, initial_channels)
        self.down_blocks = nn.ModuleList()
        current_channels = initial_channels
        for out_channels in output_channels:
            if out_channels==32:continue
            self.down_blocks.append(DownConvResBlock(current_channels, out_channels))
            current_channels = out_channels
        self.equalconv = EqualConv2d(output_channels[-1], output_channels[-1], kernel_size=3, stride=1, padding=1)
        self.linear_layers = nn.ModuleList([EqualLinear(output_channels[-1], output_channels[-1]) for _ in range(4)])
        self.final_linear = EqualLinear(output_channels[-1], dm)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        features = []
        x = self.initial_conv(x)
        x = self.down_block_0(x)
        features.append(x)
        for block in self.down_blocks:
            x = block(x)
            features.append(x)
        x = x.view(x.size(0), x.size(1), -1).mean(dim=2)
        for linear_layer in self.linear_layers:
            x = self.activation(linear_layer(x))
        x = self.final_linear(x)
        return features[::-1], x

class MotionEncoder(nn.Module):
    def __init__(self, initial_channels=64, output_channels=[64, 128, 256, 512, 512, 512], dm=32):
        super(MotionEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, initial_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.res_blocks = nn.ModuleList()
        in_channels = initial_channels
        for out_channels in output_channels:
            self.res_blocks.append(ResBlock(in_channels, out_channels))
            in_channels = out_channels
        self.equalconv = EqualConv2d(output_channels[-1], output_channels[-1], kernel_size=3, stride=1, padding=1)
        self.linear_layers = nn.ModuleList([EqualLinear(output_channels[-1], output_channels[-1]) for _ in range(4)])
        self.final_linear = EqualLinear(output_channels[-1], dm)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.equalconv(x)
        x = x.view(x.size(0), x.size(1), -1).mean(dim=2)
        for linear_layer in self.linear_layers:
            x = self.activation(linear_layer(x))
        x = self.final_linear(x)
        return x

class MotionDecoder(nn.Module):
    def __init__(self, latent_dim=32, const_dim=32):
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, const_dim, 4, 4))
        self.style_conv_layers = nn.ModuleList([
            StyledConv(const_dim, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim, upsample=True),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim, upsample=True),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 256, 3, latent_dim, upsample=True),
            StyledConv(256, 256, 3, latent_dim),
            StyledConv(256, 256, 3, latent_dim),
            StyledConv(256, 128, 3, latent_dim, upsample=True),
            StyledConv(128, 128, 3, latent_dim),
            StyledConv(128, 128, 3, latent_dim)  
        ])

    def forward(self, t):
        x = self.const.repeat(t.shape[0], 1, 1, 1)
        m1, m2, m3, m4 = None, None, None, None
        for i, layer in enumerate(self.style_conv_layers):
            x = layer(x, t)
            if i == 3:
                m1 = x
            elif i == 6:
                m2 = x
            elif i == 9:
                m3 = x
            elif i == 12:
                m4 = x
        return m1, m2, m3, m4
    
class SynthesisNetwork(nn.Module):
    def __init__(self, args, feature_dims, spatial_dims):
        super().__init__()
        self.args = args
        
        feature_dims_rev = feature_dims[::-1]
        spatial_dims_rev = spatial_dims[::-1]

        self.upconv_blocks = nn.ModuleList([
            UpConvResBlock(feature_dims_rev[i], feature_dims_rev[i+1]) for i in range(len(feature_dims_rev) - 1)
        ])
        self.resblocks = nn.ModuleList([
            ConvResBlock(feature_dims_rev[i+1]*2, feature_dims_rev[i+1]) for i in range(len(feature_dims_rev) - 1)
        ])
        
        self.transformer_blocks = nn.ModuleList()
        for i in range(len(spatial_dims_rev) - 1):
            s_dim = spatial_dims_rev[i+1]
            f_dim = feature_dims_rev[i+1]
            self.transformer_blocks.append(
                SelfAttention(args=args, dim=f_dim, resolution=(s_dim, s_dim))
            )

        self.final_conv = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_dims_rev[-1], 3*4, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Sigmoid()
        )

    def forward(self, features_align):
        x = features_align[0]
        for i in range(len(self.upconv_blocks)):
            x = self.upconv_blocks[i](x)
            x = torch.cat([x, features_align[i + 1]], dim=1)
            x = self.resblocks[i](x)
            x = self.transformer_blocks[i](x)
        return self.final_conv(x)

class IdentidyAdaptive(nn.Module):
    def __init__(self, dim_mot=32, dim_app=512, depth=4):
        super().__init__()
        self.in_layer = EqualLinear(dim_app+dim_mot, dim_app)
        self.linear_layers = nn.ModuleList([EqualLinear(dim_app, dim_app) for _ in range(depth)])
        self.final_linear = EqualLinear(dim_app, dim_mot)
        self.activation = nn.LeakyReLU(0.2)
        self.scale_activation = nn.Sigmoid()
    def forward(self, mot, app):
        x = torch.cat((mot, app), dim=-1)
        x = self.in_layer(x)
        for linear_layer in self.linear_layers:
            x = self.activation(linear_layer(x))
        out = self.final_linear(x)
        return out

class IMTRenderer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.feature_dims = [32, 64, 128, 256, 512, 512]
        self.motion_dims = self.feature_dims
        self.spatial_dims = [256, 128, 64, 32, 16, 8]

        self.dense_feature_encoder = IdentityEncoder(output_channels=self.feature_dims)
        self.latent_token_encoder = MotionEncoder(initial_channels=64, output_channels=[128, 256, 512, 512, 512])
        self.latent_token_decoder = MotionDecoder()
        self.frame_decoder = SynthesisNetwork(args, self.feature_dims, self.spatial_dims)
        self.adapt = IdentidyAdaptive()

        self.imt = nn.ModuleList()
        for dim, s_dim in zip(self.feature_dims[::-1], self.spatial_dims[::-1]):
            self.imt.append(CrossAttention(args=args, dim=dim, resolution=(s_dim, s_dim)))
        

    def decode(self, A, B, C):
        num_levels = len(self.spatial_dims)
        aligned_features = [None] * num_levels
        attention_map = None
        for i in range(num_levels):
            attention_block = self.imt[i]
            if attention_block.is_standard_attention:
                aligned_feature, attention_map = attention_block.coarse_stage(A[i], B[i], C[i])
                aligned_features[i] = aligned_feature
            else:
                aligned_feature = attention_block.fine_stage(C[i], attn=attention_map)
                aligned_features[i] = aligned_feature
        output_frame = self.frame_decoder(aligned_features)
        return output_frame
    

    def app_encode(self, x):
        f_r, id = self.dense_feature_encoder(x)
        return f_r, id
    
    def mot_encode(self, x):
        mot_latent = self.latent_token_encoder(x)
        return mot_latent
    
    def mot_decode(self, x):
        mot_map = self.latent_token_decoder(x)
        return mot_map
    
    def id_adapt(self, t, id):
        return self.adapt(t, id)
    
    def forward(self, x_current, x_reference):
        f_r, i_r = self.app_encode(x_reference)
        t_r = self.mot_encode(x_reference)
        t_c = self.mot_encode(x_current)
        ta_r = self.adapt(t_r, i_r)
        ta_c = self.adapt(t_c, i_r)
        ma_r = self.mot_decode(ta_r)
        ma_c = self.mot_decode(ta_c)
        output_frame = self.decode(ma_c, ma_r, f_r)
        return output_frame, t_c