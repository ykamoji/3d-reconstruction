import yaml
import json
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from renderer.utils import CustomObject, get_yaml_loader
from renderer.g3dr_renderer import ImportanceRenderer_extended
from eg3d.renderer import ImportanceRenderer
from renderer.model import OSGDecoder_extended, Unet
from generator_model import initialize_model


def train(config):

    results_folder = config.logging.save_dir
    render_3d = config.rendering.render
    unet_feature_dim = config.unet_feature_dim
    save_dir_checkpoints = results_folder + '/checkpoints'
    load_model = config.logging.load_model

    if render_3d:
        if config.rendering.extended_renderer:
            # print('Extending EG3D')
            renderer = ImportanceRenderer_extended()
        else:
            renderer = ImportanceRenderer()

        eg3d_decoder = OSGDecoder_extended(unet_feature_dim,
                                           options=config.rendering.triplane_renderer_config.mlp_decoder_config)
        unet_out_dim = config.unet_feature_dim * 3
    else:
        renderer = None
        eg3d_decoder = None
        unet_out_dim = 3

    model_unet = Unet(
        channels=4,
        dim=config.dim,
        out_dim=unet_out_dim,
        renderer=renderer,
        eg3d_decoder=eg3d_decoder,
        rendering_options=config.rendering.triplane_renderer_config.rendering_kwargs,
        dim_mults=(1, 2, 4, 8),
        render_3d=render_3d,
        image_size=config.image_size,
        config=config
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir_checkpoints,
        save_on_train_epoch_end=True,
        enable_version_counter=True,
        save_top_k=1,
        verbose=True,
        monitor='total_loss_step',
        every_n_train_steps=1,
        mode='min',
        filename='checkpoint-{epoch:02d}-{total_loss_step:.2f}'
    )

    trainer = Trainer(max_epochs=config.training.train_num_steps,
                      log_every_n_steps=1,
                      max_steps=2,
                      # precision="16-mixed",
                      default_root_dir=results_folder,
                      callbacks=[checkpoint_callback])

    ## TODO:: Freeze the encode and parts of decoder of the unet model
    generator3DModel = initialize_model(config, model_unet)

    model = generator3DModel()
    if load_model:
        checkpoint = torch.load(f'{config.logging.model_path}', map_location=config.device)
        if 'pytorch-lightning_version' in checkpoint.keys():
            state_dict = checkpoint['state_dict']
        else:
            state_dict = {'unet_model.' + k.partition('module.')[2]: checkpoint['model'][k] for k in checkpoint['model'].keys()}
        model.load_state_dict(state_dict)

    trainer.fit(model)


if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.load(file, get_yaml_loader())

    x = json.dumps(config)
    config = json.loads(x, object_hook=lambda d: CustomObject(**d))

    train(config)