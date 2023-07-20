# An Improved lightweight Sheep Face Detector based on RetinaFace
RetinaFace + MobileNetV3-large + SAC + CBAM with Pytorch


Pytorch version 1.7.0+ and relative torchvision are needed.

##### Data
1. Download widerface dataset


2. Organise the dataset directory as follows:

```Shell
  widerface/
    train/
      images/
      label.txt
    val/
      images/
      label.txt
    test/
      images/
      label.txt
```

## Supported environment
```
torch==1.7.0
torchvision==0.8.1
torchaudio==0.7.0
tqdm==4.61.2
tensorboard==2.5.0
terminaltables==3.1.0
tensorboardx=2.4
mmcv==1.3.9
scikit-image==0.18.2

```


## Train
```
$ train.py [-h] [data_path DATA_PATH] [--batch BATCH]
                [--epochs EPOCHS]
                [--shuffle SHUFFLE] [img_size IMG_SIZE]
                [--verbose VERBOSE] [--save_step SAVE_STEP]
                [--eval_step EVAL_STEP]
                [--save_path SAVE_PATH]
                [--depth DEPTH]
```

#### Example
For multi-gpus training, run:
```
$ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train.py --data_path /widerface --batch 32 --save_path ./out
```

#### Training log
```
---- [Epoch 39/200, Batch 400/403] ----
+----------------+-----------------------+
| loss name      | value                 |
+----------------+-----------------------+
| total_loss     | 0.09969855844974518   |
| classification | 0.09288528561592102   |
| bbox           | 0.0034053439740091562 |
| landmarks      | 0.003407923271879554  |
+----------------+-----------------------+
-------- RetinaFace Pytorch --------
Evaluating epoch 39
Recall: 0.7432201780921814
Precision: 0.906913273261629
```

