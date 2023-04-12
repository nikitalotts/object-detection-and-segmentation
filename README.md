    
# Object Detection and Segmentation

This project implements a system for segmentation and detection of abnormal clods that can form during the processing of rock in the coal industry. In order to make sure that the mechanism that processes the breed is working correctly.

The project was implemented in the 6th module of the 3rd year of [Higher IT School](https://hits.tsu.ru/), [Tomsk State University](https://www.tsu.ru/), [Tomsk](https://en.wikipedia.org/wiki/Tomsk).

## Models

The project uses two types of models: [Ultralytics YOLOv8](https://ultralytics.com/yolov8 ) (fully operational) and [U-Net](https://arxiv.org/abs/1505.04597) (the model was implemented from scratch, training and predicting methods are still in development)

##  Installation

The project does not require installation, just download, unpack the files of this repository* and install requirements**.

\* *the repository does not contain the datasets on which the model was trained, if you want to use them, then download the archive from [here](https://drive.google.com/file/d/1F05QOevK_YU8RHaHMhTfhWvKKn_m24uV/view?usp=share_link) and unpack it to the root folder of the project*

\** Run following command in shell:

``` 
$ pip install -r requirements.txt
```

## User manual

### Datasets

#### YOLO

YOLO model use datasets which consist of pair image - labels (both have the save name with different file extenstions) description and labels descriptions must be in[YOLO Darknet TXT](https://roboflow.com/formats/yolo-darknet-txt) format. Also these pairs should be spiltted into train/valid/test folders(the two last is optional) and the structute of dataset must be described in `.yaml` file

#### Example

Dataset structure:
```
dataset
│   data.yaml 
│
└───train
│   │   image1.jpg
│   │   image2.jpg
|   |   ...
│   │
│   └───labels
│       │   image1.txt
│       │   image2.txt
│       │   ...
│   
└───test
│   │   image1.jpg
│   │   image2.jpg
|   |   ...
│   │
│   └───labels
│       │   image1.txt
│       │   image2.txt
│       │   ...
|       |
└───valid
│   │   image1.jpg
│   │   image2.jpg
|   |   ...
│   │
│   └───labels
│       │   image1.txt
│       │   image2.txt
│       │   ...
```

and data.yaml file should look like this:

```
$ cat data.yaml

# pathes to train / val / test folders
train: ../train/images
val: ../valid/images
test: ../test/images

# number of classes
nc: 1 

# their names
names: ['abnormal_clod'] 
```

#### U-Net

U-Net model uses datasets in [COCO format](https://cocodataset.org/#home)

#### Example

Dataset structure:
```
dataset
│   result.json
│
└───images
│   │   image1.jpg
│   │   image2.jpg
|   |   image3.jpg
|   |   ...

```

### CLI

The system works through the following CLI commands:
1. train:
```$ python path/to/model.py train --parameter=value```

``` 
Method used to train new model

Parameters:
    dataset: str; by default: None. Path to dataset which will be used in training. If None model will use standart dataset.
    model_type: str; be default: 'YOLO'. Model type name which model will user, current possible values are 'YOLO' or 'UNET'.
    batch_size: int; by default: 4. Amount of images in batch.
    n_epochs: int; by default: 10. Amount of epochs in train cycle.

Output: new model file stored in ./outputs/user_models
```

2. evaluate: ```$ python path/to/model.py evaluate --parameter=value```

``` 
Method evaluate model's metric on given dataset's validation images

Parameters:
    dataset: str; by default: None. Path to dataset which will be used in training. If None model will use standart dataset.
    model_type: str; be default: 'YOLO'. Model type name which model will user, current possible values are 'YOLO' or 'UNET'.
    use_default_model: bool; by default: False. If true model will use standart pretrained YOLO model, else will use custom model specially trained for given task.
    model_path: str; by default: None. Path to YOLO model which will use in evaluation. If value is presented then use_default_model parameter has no effect.

Output: print in console and log in file evaluated metrics' results
```

3. convert: ```$ python path/to/model.py convert --parameter=value```

``` 
Method that get video file as input and detect and segment objects on given video

Parameters:
    video: str; by default: None. Path to video file which model will use in detection and segmentation. If None model will use standart saved video.
    model_type: str; be default: 'YOLO'. Model type name which model will user, current possible values are 'YOLO' or 'UNET'.
    process: bool; by default: False. If true model will add brighness, contrast and sharpness to every frame of the given video, else model won't change frames
    predict_on_resized: bool; by default: False. If true model will resize input frames and predict on these resized frames, else model will predict on frames of given size
    use_default_model: bool; by default: False. If true model will use standart pretrained YOLO model, else will use custom model specially trained for given task. 
    model_path: str; by default: None. Path to YOLO model which will use in evaluation. If value is presented then use_default_model parameter has no effect

Output: converted video with detected and segmented object which stored in ./outputs/{video_name}
```

4. demo: ```$ python path/to/model.py demo --parameter=value```

``` 
Method that get video file as input and start stream on object detection and segmentation on video.

Parameters:
    video: str; by default: None. Path to video file which model will use in detection and segmentation.  If None model will use standart saved video.
    model_type: str; be default: 'YOLO'. Model type name which model will user, current possible values are 'YOLO' or 'UNET'.
    process: bool; by default: True. If true model will add brighness, contrast and sharpness to every frame of the given video, else model won't change frames
    use_default_model: bool; by default: False. If true model will use standart pretrained YOLO model, else will use custom model specially trained for given task. 
    model_path: str; by default: None. Path to YOLO model which will use in evaluation. If value is presented then use_default_model parameter has no effect

Output: return nothing
```

### Example of work

Here you can see a fragment of the video processed by the system:
![](./examples/example.gif)

### Sources
* [YOLOv8 Docs](https://docs.ultralytics.com/)
* [Complete YOLO v8 Custom Object Detection Tutorial | Windows & Linux](https://www.youtube.com/watch?v=gRAyOPjQ9_s)
* [Instance segmentation YOLO v8 | Opencv with Python tutorial](https://www.youtube.com/watch?v=cHOOnb_o8ug)

### Author
*Nikita Lotts, 3rd grade student in Tomsk State University (Tomsk)*