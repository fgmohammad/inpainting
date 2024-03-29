# Inpainting Cosmological Maps with Deep Learning 
https://arxiv.org/abs/2109.07070

## Requirements  
Pytorch 1.9  
python 3.8  
numpy 1.20  
opencv 4.5  
astropy 4.2  


## masks.py
Creates Binary masks of various shapes and sizes

## Usage
### Training  

  `$ python main_training.py --image_root your_image_root --masks_root your_mask_root --save_dir your_save_dir --batch_size batch_size`  

### Prediction  
  `$ python prediction.py --image_path your_image_path --mask_path your_mask_path --save_dir your_save_dir --model_path model_path`

### Results
![results](/plots/maps_github.png)
