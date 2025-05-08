# Generative 3D Reconstruction

### Presentation

[![YouTube](https://img.icons8.com/color/48/000000/youtube-play.png)](https://www.youtube.com/watch?v=wzeR1Mai3MQ)

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
  - If using mps (Apple GPU), some operations are not compatible. To allow fallback to CPU add the below environment 
  variable,
    ```angular2html 
    PYTORCH_ENABLE_MPS_FALLBACK=1
    ```
- **batch_size** should be updated according to available GPU / CPU memory. Default 1.
- Set **load_model** false for training with newly initialized network.
- Update **model_path** for the pretrained model.
- For generating 3D model videos, update the **image_path** with the directory containing source images.
- For running visualization file, need to install additional library: **ffmpeg**
  - For Mac OS: 
    ```angular2html 
      brew install ffmpeg
      ```
  - For Linux 
     ```angular2html 
      sudo apt-get install ffmpeg
      ```
    ```angular2html 
      sudo apt install ffmpeg
      ```
  - For Windows
    - Download the <a href='https://www.ffmpeg.org/download.html#build-windows'> FFmpeg package </a> from the official website
    - Choose the Windows builds from gyan.dev. This will redirect you to the gyan.dev website.
    - Select the ffmpeg-git-full.7z version.
    - Once downloaded, right-click the FFmpeg folder and select Extract files.
    - Once done, open the extracted folder and copy and paste all the EXE files from bin to the root folder of your hard drive. For example, create a separate folder on the Local Disk (C:) to store all the files.
    - Type “environment properties” on the search tab and click Open.This will open the System Properties window. Go to the Advanced tab and choose Environment Variables…
    - Go to the System variables section. Select Path from the list and click Edit.
    - Choose New and add the FFmpeg path of the folder you have created previously to store the EXE files.
    - Once done, click OK to save your changes. This will close the Edit environment variable window.
    - Run the following command to verify that FFmpeg is installed:
     ```angular2html 
      ffmpeg
      ```
<hr>

### Pre trained model

Pretrained, fine-tuned and optimized can be found here:
https://drive.google.com/drive/folders/1Q8DHDj4rQxuuR2A4scv5xRPf1nnNQ-eb?usp=sharing
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
