# PaDiM-EfficientNet

There are two differences from the existing [PaDiM code](https://github.com/youngjae-avikus/PaDiM-EfficientNet/tree/master). 

1. Edit the heat map display to match the color bar so that max = 1 and min = 0 (not max, min = image highest, lowest score)
2. Create a separate inference file for new data (only images, without ground truths and labels) with manual thresholding
3. Inference scores normalized using bad+good test score max/min which were set manually, not using inference score max/min
4. Fix the size of predicted mask so that smaller anomaly could be detected

## Requirement


* [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)



## Datasets
MVTec AD datasets : Download from [MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

## Usage
```bash
# training and testing
train_test_screw.py
# infer
screw_inference.py
```


## Reference
[1] https://github.com/youngjae-avikus/PaDiM-EfficientNet/tree/master
