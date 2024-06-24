This code is 

# Pulp Segmentation Framework
Pulp Segmentation Training Framework for 3D CBCT Scans

This code is inspired from the ToothFairy Challenge (MICCAI 2023) and the source code provided in AImageLab [alveolar_canal](https://github.com/AImageLab-zip/alveolar_canal)

## Hardware Setup
| Component | Specification             |
| --------- | ------------------------- |
| CPU       | Intel i7 13700K - 24 core |
| GPU       | Nvidia 4090-RTX 24GB      |
| RAM       | 32 GB DDR4                |
| SSD       | 1TB M2                    |


## Installation
```
# Create python virtual environment
virtualenv venv -p $(which python3)

# Activate virtual environment
source venv/local/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Training
```
python main.py --config configs/segmentation.yml
```

## Dataset
You can access the infered data using our model using this [link](http://eng.staff.alexu.edu.eg/staff/mtorki/Research/Data/JDR_Files/)
