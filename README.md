# Generative 3D Reconstruction

### Requirements

```angular2html
pip3 install -r requirements.txt
```
<hr>

### Configurations
<hr>
In config.yaml,<br>

-   Update **dataset_folder** for training or add an environment variable DATASET_PATH.
- Update **device** to use cuda, cpu or mps. 
  - Note: use cpu only for visualization. Training on cpu will crash! 
- **batch_size** should be updated according to available GPU / CPU memory. Default 1.
- Set **load_model** false for training with newly initialized network.
- Update **model_path** for the pretrained model.
- For generating 3D model videos, update the **image_path** with the directory containing source images.

<hr>

### Pre trained model

Pretrained f_trigen on imagenet can be found here: https://drive.google.com/file/d/1Bg5k3IYquph-cZbWJVW0A4kyyd-t7n-d/view

To train, run the below command

```bash 
python3 train.py 
```
<hr>
To generate 3D models, run the below command

```bash 
python3 visualize.py 
```
<hr>