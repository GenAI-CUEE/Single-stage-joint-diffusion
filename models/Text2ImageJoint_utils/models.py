from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer


def get_DDPMscheduler(config):
    return DDPMScheduler.from_pretrained(config["pretrained_model"], subfolder="scheduler")


def get_CLIPTokenizer(config):
    return CLIPTokenizer.from_pretrained(config["pretrained_model"], subfolder="tokenizer", revision=None)


def get_CLIPTextModel(pretrained, config):
    if pretrained:
        return CLIPTextModel.from_pretrained(config["pretrained_model"], subfolder="text_encoder")

    else:
        configuration = CLIPTextConfig(vocab_size=49408,
                                       hidden_size=8,
                                       intermediate_size=64,
                                       projection_dim=8,
                                       num_hidden_layers=1,
                                       num_attention_heads=1,
                                       max_position_embeddings=77,
                                       pad_token_id=1,
                                       bos_token_id=49406,
                                       eos_token_id=49407
                                       )
        return CLIPTextModel(configuration)


def get_vae(pretrained, config):
    if pretrained:
        return AutoencoderKL.from_pretrained(config["pretrained_model"], subfolder="vae")
    else:
        return AutoencoderKL()


def get_unet(pretrained, config):
    if pretrained:
        return UNet2DConditionModel.from_pretrained(config["pretrained_model"], subfolder="unet")
    else:
        return UNet2DConditionModel(
                sample_size=64,  # the target image resolution
                in_channels=4,  # the number of input channels, 3 for RGB images
                out_channels=4,  # the number of output channels
                layers_per_block=2,  # how many ResNet layers to use per UNet block
                encoder_hid_dim = config["unet_encoder_hid_dim"],
                cross_attention_dim = 128,
                block_out_channels=(128, 256, 512, 512),
                # the number of output channels for each UNet block
                down_block_types=(
                    "CrossAttnDownBlock2D",  # a regular ResNet downsampling block
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",  # a regular ResNet upsampling block
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                ),
            )
