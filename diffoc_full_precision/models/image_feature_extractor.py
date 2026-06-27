import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from models.model_utils import adapt_input_conv, resize_pos_embed, init_weights, padding, unpadding, GeMPooling
from models.stems import PatchEmbedding, ConvStem
from models.decoders import DecoderLinear
# DINOv2
from models.layers import Mlp, NestedTensorBlock as Block

# Modified from https://github.com/valeoai/rangevit/blob/main/models/rangevit.py

logger = logging.getLogger(__name__)


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
            self,
            image_size,
            patch_size,
            n_layers,
            d_model,
            d_ff,
            n_heads,
            dropout=0.1,
            drop_path_rate=0.0,
            channels=3,
            ls_init_values=None,
            patch_stride=None,
            conv_stem='none',
            stem_base_channels=32,
            stem_hidden_dim=None,
            n_cls=1
    ):
        super().__init__()

        self.conv_stem = conv_stem

        if self.conv_stem == 'none':
            self.patch_embed = PatchEmbedding(
                image_size,
                patch_size,
                patch_stride,
                d_model,
                channels, )
        else:  # in this case self.conv_stem = 'ConvStem'
            assert patch_stride == patch_size  # patch_size = patch_stride if a convolutional stem is used

            self.patch_embed = ConvStem(
                in_channels=channels,
                base_channels=stem_base_channels,
                img_size=image_size,
                patch_stride=patch_stride,
                embed_dim=d_model,
                flatten=True,
                hidden_dim=stem_hidden_dim)

        self.patch_size = patch_size
        self.PS_H, self.PS_W = patch_size
        self.patch_stride = patch_stride
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls
        self.image_size = image_size

        mlp_ratio = 4
        qkv_bias = True
        proj_bias = True
        ffn_bias = True

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU
        ffn_layer = Mlp
        init_values = 1.0

        blocks_list = [
            Block(
                dim=d_model,
                num_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(n_layers)
        ]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.num_patches + 1, d_model))


        self.blocks = nn.ModuleList(blocks_list)
        self.chunked_blocks = False

        self.norm = norm_layer(d_model)
        self.head = nn.Identity()


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_grid_size(self, H, W):
        return self.patch_embed.get_grid_size(H, W)

    def prepare_tokens(self, x):
        B, _, W, H = x.shape
        x, skip = self.patch_embed(x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        pos_embed = self.pos_embed
        num_extra_tokens = 1

        if x.shape[1] != pos_embed.shape[1]:
            grid_H, grid_W = self.get_grid_size(H, W)
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (grid_H, grid_W),
                num_extra_tokens,
            )

        x = x + pos_embed
        x = self.dropout(x)

        return x, skip

    def forward(self, im, return_features=False):
        x, skip = self.prepare_tokens(im)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.head(x)

        return x, skip


def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    model_cfg.pop('backbone')
    mlp_expansion_ratio = 4
    model_cfg['d_ff'] = mlp_expansion_ratio * model_cfg['d_model']

    new_patch_size = model_cfg.pop('new_patch_size')
    new_patch_stride = model_cfg.pop('new_patch_stride')

    if (new_patch_size is not None):
        if new_patch_stride is None:
            new_patch_stride = new_patch_size
        model_cfg['patch_size'] = new_patch_size
        model_cfg['patch_stride'] = new_patch_stride

    model = VisionTransformer(**model_cfg)

    return model


def create_decoder(decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop('name')
    decoder = DecoderLinear(**decoder_cfg)

    return decoder, name


def create_rangevit(model_cfg):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop('decoder')

    encoder = create_vit(model_cfg)

    decoder, name = create_decoder(decoder_cfg)

    model = RangeViT(encoder, decoder, n_cls=model_cfg['n_cls'])

    return model


class RangeViT(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls

        # patch 的大小和 stride（通常来自 ViT 或 Conv stem）
        self.patch_size = encoder.patch_size
        self.patch_stride = encoder.patch_stride

        # backbone + decoder
        self.encoder = encoder
        self.decoder = decoder
        # self.pool = GeMPooling()  # 原始代码中可能用于最后特征池化，此处未启用

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        获取 encoder 和 decoder 中声明的 no_weight_decay 参数列表。
        作用：告诉 Optimizer 某些参数（例如 LayerNorm、pos_embedding）
             不需要 weight decay。
        """

        def append_prefix_no_weight_decay(prefix, module):
            # module.no_weight_decay() 返回参数名集合，这里给每个参数名前面加 prefix
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        # 合并 encoder 和 decoder 的 no_weight_decay 参数
        nwd_params = append_prefix_no_weight_decay('encoder.', self.encoder).union(
             append_prefix_no_weight_decay('decoder.', self.decoder)
        )
        return nwd_params


    def forward(self, im):
        """
        输入:
            im: [B*N, C=32, H=720, W=5]  —— Range Image 格式（例子）
        输出:
            x:    [B*N, d]        —— 全局特征，用于后续定位/匹配
            feats [B*N, C, H, W]  —— 上采样后的 dense feature map
        """

        # 原图尺寸（去 padding 用）
        H_ori, W_ori = im.size(2), im.size(3)

        # 1. 根据 patch_size 做 padding（通常要 pad 成 patch 整除）
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        # 2. backbone 前向
        #    return_features=True 表示同时返回 skip（多尺度特征，用于 decoder）
        x, skip = self.encoder(im, return_features=True)

        # x shape: [B, 1 + HW/P^2, d]
        # skip: encoder 中间层输出，多用于 decoder 做 feature fusion

        # 3. 移除 CLS token，因为 decoder 不需要 CLS（只要 patch token）
        num_extra_tokens = 1  # 一般是 1 个 CLS token
        # x的尺寸：54,256,384
        x = x[:, num_extra_tokens:]   # x -> 只剩下所有 patch token

        # 4. decoder 解码 —— 输出 pred mask: 54,256,1 和 dense features: 54,1,8,32
        pred_mask, feats = self.decoder(x, (H, W), skip)
        # pred_mask: [B, N_patch, 1] 或 reshape 后 [B, H', W']
        # feats: decoder 输出的 dense 特征图

        # 5. mask 用 sigmoid 做归一化,形状不变
        pred_mask = torch.sigmoid(pred_mask)

        # 6. 将 decoder 输出的 dense features 上采样回 padding 后的尺寸：54，1，32，512
        feats = F.interpolate(feats, size=(H, W), mode='bilinear')

        # 7. 去掉 padding —— 恢复原图 H_ori × W_ori 的 dense 特征图：54，1，32，512
        feats = unpadding(feats, (H_ori, W_ori))

        # 8. 对 patch token 做 mask 选择性加权
        #    x 的 shape 为 [B, N_patch, d] 56，256，384
        #    pred_mask broadcast 成同维度后：x + x * pred_mask = x*(1 + mask)
        #    即使 mask 更强的区域贡献更大
        x = (x + x * pred_mask).mean(1)  # 对所有 patch 求平均，得到全局 embedding

        return x, feats # x尺寸：54,384   //   feats尺寸：54,1,32,512  //为什么B是54——batchsize=18, 训练时timestepx3



class ImageFeatureExtractor(nn.Module):
    def __init__(
            self,
            backbone: str = "vit_base_patch16_384",
            freeze=False,
            in_channels=5,
            new_patch_size=(4, 16),
            new_patch_stride=(4, 16),
            conv_stem='ConvStem',  # 'none' or 'ConvStem'
            stem_base_channels=32,
            D_h=256,  # hidden dimension of the stem
            image_size=(32, 512),
            decoder='up_conv',
            pretrained_path="dino_vitbase16_pretrain.pth",
            reuse_pos_emb=True,
            reuse_patch_emb=False,
            n_cls=1
    ):
        super().__init__()

        if backbone == 'vit_small_patch16_384':
            n_heads = 6
            n_layers = 12
            patch_size = 16
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 384
        elif backbone == 'vit_base_patch16_384':
            n_heads = 12
            n_layers = 12
            patch_size = 16
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 768
        elif backbone == 'vit_large_patch16_384':
            n_heads = 16
            n_layers = 24
            patch_size = 16
            dropout = 0.0
            drop_path_rate = 0.1
            d_model = 1024
        else:
            raise NameError('Not known ViT backbone.')

        # Decoder config
        if decoder == 'linear':
            decoder_cfg = {'name': 'linear',
                           'patch_size': new_patch_size,
                            'patch_stride': new_patch_stride,
                            'd_encoder': d_model,
                            'n_cls': n_cls}
        elif decoder == 'up_conv':
            decoder_cfg = {
                'name': 'up_conv',
                'patch_size': new_patch_size,
                'patch_stride': new_patch_stride,
                'd_encoder': d_model,
                'n_cls': n_cls,
                'd_decoder': 64,  # hidden dim of the decoder
                'scale_factor': new_patch_size,  # scaling factor in the PixelShuffle layer
                'skip_filters': 256 }  # channel dim of the skip connection (between the convolutional stem and the up_conv decoder)

        # ViT encoder and stem config
        net_kwargs = {
            'backbone': backbone,
            'd_model': d_model,  # dim of features
            'decoder': decoder_cfg,
            'drop_path_rate': drop_path_rate,
            'dropout': dropout,
            'channels': in_channels,  # nb of channels for the 3D point projections
            'image_size': image_size,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'patch_size': patch_size,  # old patch size for the ViT encoder
            'new_patch_size': new_patch_size,  # new patch size for the ViT encoder
            'new_patch_stride': new_patch_stride,  # new patch stride for the ViT encoder
            'conv_stem': conv_stem,
            'stem_base_channels': stem_base_channels,
            'stem_hidden_dim': D_h,
            'n_cls': n_cls  # moving objects / static objects
        }

        # Create RangeViT model
        self.rangevit = create_rangevit(net_kwargs)
        old_state_dict = self.rangevit.state_dict()
        self._output_dim = d_model

        # Loading pre-trained weights in the ViT encoder
        if pretrained_path is not None:
            path = Path(pretrained_path)
            if not path.is_file():
                repo_root = Path(__file__).resolve().parents[2]
                alt_path = repo_root / pretrained_path
                if alt_path.is_file():
                    path = alt_path
                else:
                    raise FileNotFoundError(f"Pretrained weights not found at {pretrained_path} or {alt_path}")
            print(f'Loading pretrained parameters from {path}')
            if pretrained_path == 'timmImageNet21k':
                vit_imagenet = timm.create_model(backbone, pretrained=True)  # .cuda()
                pretrained_state_dict = vit_imagenet.state_dict()  # nb keys: 152
                all_keys = list(pretrained_state_dict.keys())
                for key in all_keys:
                    pretrained_state_dict['encoder.' + key] = pretrained_state_dict.pop(key)
            else:
                pretrained_state_dict = torch.load(path, map_location='cpu')
                if 'model' in pretrained_state_dict:
                    pretrained_state_dict = pretrained_state_dict['model']
                elif 'pos_embed' in pretrained_state_dict.keys():
                    all_keys = list(pretrained_state_dict.keys())
                    for key in all_keys:
                        pretrained_state_dict['encoder.' + key] = pretrained_state_dict.pop(key)

            # Reuse pre-trained positional embeddings
            if reuse_pos_emb:
                pos_key = 'encoder.pos_embed'
                alt_pos_key = 'pos_embed'
                if pos_key not in pretrained_state_dict and alt_pos_key in pretrained_state_dict:
                    pretrained_state_dict[pos_key] = pretrained_state_dict.pop(alt_pos_key)
                if pos_key in pretrained_state_dict:
                    print('Reusing positional embeddings.')
                    gs_new_h = int((image_size[0] - new_patch_size[0]) // new_patch_stride[0] + 1)
                    gs_new_w = int((image_size[1] - new_patch_size[1]) // new_patch_stride[1] + 1)
                    num_extra_tokens = 1
                    resized_pos_emb = resize_pos_embed(pretrained_state_dict[pos_key],
                                                       grid_old_shape=None,
                                                       grid_new_shape=(gs_new_h, gs_new_w),
                                                       num_extra_tokens=num_extra_tokens)
                    pretrained_state_dict[pos_key] = resized_pos_emb
                else:
                    print(f'Warning: positional embedding not found in checkpoint, skip reuse.')
            else:
                pretrained_state_dict.pop('encoder.pos_embed', None)  # remove positional embeddings

            # Reuse pre-trained patch embeddings
            if reuse_patch_emb:
                assert conv_stem == 'none'  # no patch embedding if a convolutional stem is used
                print('Reusing patch embeddings.')

                bias_key = 'encoder.patch_embed.proj.bias'
                weight_key = 'encoder.patch_embed.proj.weight'
                if bias_key in pretrained_state_dict and weight_key in pretrained_state_dict:
                    assert old_state_dict[bias_key].shape == pretrained_state_dict[bias_key].shape
                    old_state_dict[bias_key] = pretrained_state_dict[bias_key]

                    _, _, gs_new_h, gs_new_w = old_state_dict[weight_key].shape
                    reshaped_weight = adapt_input_conv(in_channels,
                                                       pretrained_state_dict[weight_key])
                    reshaped_weight = F.interpolate(reshaped_weight, size=(gs_new_h, gs_new_w), mode='bilinear')
                    pretrained_state_dict[weight_key] = reshaped_weight
                else:
                    print('Warning: patch embedding not found in checkpoint, skip reuse.')
            else:
                pretrained_state_dict.pop('encoder.patch_embed.proj.weight', None)  # remove patch embedding layers
                pretrained_state_dict.pop('encoder.patch_embed.proj.bias', None)  # remove patch embedding layers

            # Delete the pre-trained weights of the decoder
            decoder_keys = []
            for key in pretrained_state_dict.keys():
                if 'decoder' in key:
                    decoder_keys.append(key)
            for decoder_key in decoder_keys:
                del pretrained_state_dict[decoder_key]

            msg = self.rangevit.load_state_dict(pretrained_state_dict, strict=False)
            print(f'{msg}')

        if freeze:
            print('==> Freeze the ViT encoder (without the pos_embed and stem)')
            for param in self.feature_extractor.blocks.parameters():
                param.requires_grad = False

            self.feature_extractor.norm.weight.requires_grad = False
            self.feature_extractor.norm.bias.requires_grad = False


    def get_output_dim(self):
        return self._output_dim

    def forward(self, *args):
        return self.rangevit(*args)
