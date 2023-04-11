import os
import cv2
import moviepy
import shutil
import numpy as np
import ultralytics
import moviepy.video.io.ImageSequenceClip
from ultralytics import YOLO
from matplotlib import image as img
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
from datetime import datetime
import src.models.UnetModel
from src.models.UnetModel import UnetModel
from src.models.YoloModel import YoloModel
import fire
import torch, gc
from src.logger import logger


class ODaSModel:
    def __init__(self):
        logger.info('started ODaSModel instance init')
        self.model = None
        logger.info('finished ODaSModel instance init')

    def setup_env(self):
        logger.info('started venv setup')
        os.environ['PATH_TO_VIDEO'] = 'C:/Users/Acer/Machine Learning/ODaS/clodding_train.mp4'
        os.environ['PROJECT_FOLDER']  = os.path.dirname(os.path.dirname(os.path.abspath('__file__'))) + '\\ODaS'
        os.environ['INPUT_FRAMES_FOLDER'] = os.path.join(os.environ['PROJECT_FOLDER'], '.input_frames')
        os.environ['PROCESSED_FRAMES_FOLDER'] = os.path.join(os.environ['PROJECT_FOLDER'], '.processed_frames')
        os.environ['OUTPUT_FOLDER'] = os.path.join(os.environ['PROJECT_FOLDER'], 'outputs')
        os.environ['FRAMES_EXTENSION'] = '.jpg'
        os.environ['CONVERT_BRIGHTNESS'] = str(100)
        os.environ['CONVERT_CONTRAST'] = str(65)
        os.environ['CONVERT_SHARPENING_CYCLE'] = str(1)
        os.environ['TASK_TYPE'] = 'segment'
        os.environ['N_EPOCHS'] = str(10)
        os.environ['BATCH_SIZE'] = str(4)
        os.environ['TRAIN_IMAGE_SIZE'] = str(640)
        os.environ['IMAGE_SIZE'] = str(640)
        os.environ['FPS'] = '5.0'
        os.environ['YOLO_MODEL_PATH'] = os.path.join(os.environ['PROJECT_FOLDER'], './src/models_store/yolo/seg_640_n.pt')
        os.environ['UNET_MODEL_PATH'] = os.path.join(os.environ['PROJECT_FOLDER'], './src/models_store/unet/unet.pth')
        os.environ['YOLO_PRETRAINED_MODEL_TYPE'] = 'yolov8n-seg.pt'
        os.environ['YOLO_PRETRAINED_MODEL_PATH'] = os.path.join(os.environ['PROJECT_FOLDER'], f'./src/models_store/yolo/{os.environ["YOLO_PRETRAINED_MODEL_TYPE"]}')
        os.environ['UNET_PRETRAINED_MODEL_PATH'] = os.path.join(os.environ['PROJECT_FOLDER'], f'./src/models_store/unet/unet_trained.pth')
        os.environ['YOLO_TRAIN_DATA_PATH'] = os.path.join(os.environ['PROJECT_FOLDER'], './datasets/yolo/data.yaml')
        os.environ['YOLO_RUNS_FOLDER_PATH'] = os.path.join(os.environ['PROJECT_FOLDER'], './runs')
        os.environ['YOLO_CUSTOM_TRAIN_MODEL_PATH'] = os.path.join(os.environ['YOLO_RUNS_FOLDER_PATH'], f'./{os.environ["TASK_TYPE"]}/train/weights/best.pt')
        os.environ['YOLO_CUSTOM_TRAIN_MODELS_FOLDER_PATH'] = os.path.join(os.environ['OUTPUT_FOLDER'], './user_models')
        os.environ['PREDICTED_DATA_PATH'] = os.path.join(os.environ['PROJECT_FOLDER'], 'runs/segment/predict')
        os.environ['PREDICTED_LABELS_PATH'] = os.path.join(os.environ['PROJECT_FOLDER'], 'runs/segment/predict/labels')
        os.environ['UNET_TRAIN_DATA_PATH'] = os.path.join(os.environ['PROJECT_FOLDER'], './datasets/coco/result.json')
        os.environ['TEMP_FOLDER'] = os.path.join(os.environ['PROJECT_FOLDER'], './.temp')
        os.environ['TEMP_IMAGE_NAME'] = 'frame'
        os.environ['TEMP_IMAGE_FILE'] = os.path.join(os.environ['TEMP_FOLDER'], f"./{os.environ['TEMP_IMAGE_NAME']}{os.environ['FRAMES_EXTENSION']}")
        os.environ['TEMP_LABEL_FILE'] = os.path.join(os.environ['PREDICTED_LABELS_PATH'], f"./{os.environ['TEMP_IMAGE_NAME']}.txt")
        os.environ['PREDICTED_IMAGE_PATH'] = os.path.join(os.environ['PREDICTED_DATA_PATH'], f"./{os.environ['TEMP_IMAGE_NAME']}{os.environ['FRAMES_EXTENSION']}")

        isExist = os.path.exists(os.environ['INPUT_FRAMES_FOLDER'])
        if not isExist:
            os.makedirs(os.environ['INPUT_FRAMES_FOLDER'])
            logger.info(f"created folder {os.environ['INPUT_FRAMES_FOLDER']}")
        isExist = os.path.exists(os.environ['OUTPUT_FOLDER'])
        if not isExist:
            os.makedirs(os.environ['OUTPUT_FOLDER'])
            logger.info(f"created folder {os.environ['OUTPUT_FOLDER']}")
        isExist = os.path.exists(os.environ['PROCESSED_FRAMES_FOLDER'])
        if not isExist:
            os.makedirs(os.environ['PROCESSED_FRAMES_FOLDER'])
            logger.info(f"created folder {os.environ['PROCESSED_FRAMES_FOLDER']}")
        isExist = os.path.exists(os.environ['TEMP_FOLDER'])
        if not isExist:
            os.makedirs(os.environ['TEMP_FOLDER'])
            logger.info(f"created folder {os.environ['TEMP_FOLDER']}")
        logger.info('finished venv setup')

    def __get_capture(self, path_to_video: str):
        logger.info('started getting capture')
        capture = cv2.VideoCapture(path_to_video)
        os.environ['VIDEO_EXTENSION'] = os.path.splitext(path_to_video)[1]  #
        os.environ['VIDEO_NAME'] = os.path.basename(path_to_video)
        os.environ['FPS'] = str(capture.get(cv2.CAP_PROP_FPS))
        # os.environ['FPS'] = str(6)
        os.environ['FRAMES_AMOUNT'] = str(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        os.environ['VIDEO_FRAME_WIDTH'] = str(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        os.environ['VIDEO_FRAME_HEIGHT'] = str(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        logger.info('finished getting capture')
        return capture

    def __split_video_into_frames(self, path_to_video):
        logger.info('started splitting video into frames')
        if not os.path.exists(path_to_video):
            raise FileExistsError('Invalid path to video file')
        capture = self.__get_capture(path_to_video)
        frame_number = 0
        while (True):
            success, frame = capture.read()
            if success:
                cv2.imwrite(os.path.join(os.environ['INPUT_FRAMES_FOLDER'], f'{frame_number}{os.environ["FRAMES_EXTENSION"]}'), frame)
            else:
                break
            frame_number = frame_number+1
        capture.release()
        logger.info('successfully finished splitting video into frames')

    def __convert_frames_into_video(self):
        logger.info('started converting video into frames')
        if not os.path.exists(os.environ['PREDICTED_DATA_PATH']):
            raise NotADirectoryError("No such dir")

        files_in_folder = [f for f in os.listdir(os.environ['PREDICTED_DATA_PATH']) if f.endswith(os.environ['FRAMES_EXTENSION'])]

        frames = [os.path.join(os.environ['PREDICTED_DATA_PATH'],img)
                  for img in sorted(files_in_folder, key=lambda x: int(os.path.splitext(x)[0]))]

        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames, float(os.environ['FPS']))
        save_path = os.path.join(os.environ['OUTPUT_FOLDER'], os.environ['VIDEO_NAME'])
        clip.write_videofile(save_path, audio=False)
        logger.info('successfully finished converting video into frames')
        return save_path

    def clear_cache(self):
        logger.info('started deleting temp files and folders')
        if os.path.exists(os.environ['INPUT_FRAMES_FOLDER']):
            shutil.rmtree(os.environ['INPUT_FRAMES_FOLDER'])

        if os.path.exists(os.environ['PROCESSED_FRAMES_FOLDER']):
            shutil.rmtree(os.environ['PROCESSED_FRAMES_FOLDER'])

        if os.path.exists(os.environ['TEMP_FOLDER']):
            shutil.rmtree(os.environ['TEMP_FOLDER'])

        if os.path.exists(os.environ['YOLO_RUNS_FOLDER_PATH']):
            shutil.rmtree(os.environ['YOLO_RUNS_FOLDER_PATH'])
        logger.info('successfully finished deleting temp files and folders')

    def __apply_brightness_contrast_sharpening(self, input_img, brightness = 0, contrast = 0, sharp_cyc=1, prod=False, process=True, resize=True):
        logger.info('started adding filters to image')
        if not process:
            if resize:
                image = cv2.resize(input_img.copy(), (int(os.environ['IMAGE_SIZE']), int(os.environ['IMAGE_SIZE'])), interpolation = cv2.INTER_AREA)
            else:
                image = input_img
                os.environ['IMAGE_SIZE'] = str(max(int(os.environ['VIDEO_FRAME_WIDTH']), int(os.environ['VIDEO_FRAME_HEIGHT'])))
            return image
        else:
            brightness = self.__map(brightness, 0, 510, -255, 255)
            contrast = self.__map(contrast, 0, 254, -127, 127)

            if brightness != 0:
                if brightness > 0:
                    shadow = brightness
                    highlight = 255
                else:
                    shadow = 0
                    highlight = 255 + brightness
                alpha_b = (highlight - shadow)/255
                gamma_b = shadow

                buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
            else:
                buf = input_img.copy()

            if contrast != 0:
                f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
                alpha_c = f
                gamma_c = 127*(1-f)

                buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

            if sharp_cyc != 0:
                buf = self.__apply_sharpen(buf, sharp_cyc)

            if not prod:
                cv2.putText(buf,'B:{},C:{}'.format(brightness,contrast),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if resize:
                buf = cv2.resize(buf, (int(os.environ['IMAGE_SIZE']), int(os.environ['IMAGE_SIZE'])), interpolation = cv2.INTER_AREA)
            else:
                os.environ['IMAGE_SIZE'] = str(max(int(os.environ['VIDEO_FRAME_WIDTH']), int(os.environ['VIDEO_FRAME_HEIGHT'])))

            logger.info('successfully added filters to image')
            return buf

    def __apply_sharpen(self, img, n=1):
        logger.info('started sharpening of image')
        sharpen_filter=np.array([[-1,-1,-1],
                                 [-1,9,-1],
                                 [-1,-1,-1]])
        sharp_image = img
        for i in range(n):
            sharp_image=cv2.filter2D(sharp_image, -1, sharpen_filter)

        # another way
        #     unsharped = cv2.addWeighted(img, 1.5, smoothed, -0.5, 0)
        logger.info('successfully sharpen image')
        return sharp_image

    def __change_frames(self, process=True, resize=True):
        logger.info('started changing frames method')
        input_frames_folder = os.environ['INPUT_FRAMES_FOLDER']
        output_dir = os.environ['PROCESSED_FRAMES_FOLDER']

        if not os.path.exists(output_dir):
            logger.error(f'Directory doesn\'t exist {output_dir}')
            raise NotADirectoryError("No such dir")

        for (i, image) in enumerate(sorted(os.listdir(input_frames_folder), key=lambda x: int(os.path.splitext(x)[0]))):
            # check if the image ends with png or jpg or jpeg
            if (image.endswith(os.environ['FRAMES_EXTENSION'])):
                img = cv2.imread(os.path.join(input_frames_folder, image), cv2.IMREAD_COLOR)
                cv2.imwrite(os.path.join(output_dir, f'{i}.jpg'), self.__apply_brightness_contrast_sharpening(img, int(os.environ['CONVERT_BRIGHTNESS']) + 255,
                                                                                                            int(os.environ['CONVERT_CONTRAST']) + 127,
                                                                                                            int(os.environ['CONVERT_SHARPENING_CYCLE']), prod=True,
                                                                                                            process=process, resize=resize))
        logger.info('changing frames method successfully executed')

    def __map(self, x, in_min, in_max, out_min, out_max):
        return int((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)

    # def proceed_video(self, path_to_video, change_video=True):
    #     logger.info('started proceed_video method')
    #     self.__split_video_into_frames(path_to_video)
    #     if change_video:
    #         self.__change_frames()
    #
    #     # predict
    #     self.__convert_frames_into_video(change_video)
    #     self.clear_cache()
    #     logger.info('proceed_video method successfully executed')

    def __get_dataset_yaml_file(self, path_to_dataset):
        logger.info('started __get_dataset_yaml_file method')
        yaml_files = [f for f in os.listdir(path_to_dataset) if f.endswith('.yaml')]
        if yaml_files:
            data_file = os.path.join(path_to_dataset, yaml_files[0])
            return data_file
        else:
            logger.error("No .yaml dataset file")
            raise FileNotFoundError('YOLO model requares .yaml dataset description')
        logger.info('__get_dataset_yaml_file method successfully executed')

    def __save_model(self):
        logger.info('started __save_model method')
        if not os.path.exists(os.environ['YOLO_CUSTOM_TRAIN_MODELS_FOLDER_PATH']):
            os.mkdir(os.environ['YOLO_CUSTOM_TRAIN_MODELS_FOLDER_PATH'])
        if self.model_type == 'YOLO':
            if os.path.exists( os.environ['YOLO_CUSTOM_TRAIN_MODEL_PATH']):
                new_path = str('_'.join([self.model_type, str(datetime.now().timestamp()).split(".")[0]]))+f'.pt'
                os.rename(os.environ['YOLO_CUSTOM_TRAIN_MODEL_PATH'], os.path.join(os.environ['YOLO_CUSTOM_TRAIN_MODELS_FOLDER_PATH'], new_path))
            else:
                raise FileNotFoundError("trained model haven't saved")
        elif self.model_type == 'UNET':
            pass
        else:
            raise NotImplementedError("attempt to save wrong model type")
        logger.info('__save_model method successfully executed')

    def train(self, path_to_dataset: str = None, model_type: str = 'YOLO', batch_size: int = 4, n_epochs: int = 10):
        logger.info('started train method')
        if model_type == 'YOLO':
            if self.model is None:
                self.load_model('YOLO', use_default_model=True)
            if path_to_dataset is None:
                path_to_dataset = os.environ['YOLO_TRAIN_DATA_PATH']
            elif path_to_dataset != os.environ['YOLO_TRAIN_DATA_PATH']:
                path_to_dataset = self.__get_dataset_yaml_file(path_to_dataset)
            # print(os.environ['YOLO_PRETRAINED_MODEL_PATH'], os.environ['TASK_TYPE'])
            results = self.model.train(data=path_to_dataset, imgsz=int(os.environ['TRAIN_IMAGE_SIZE']),
                                       epochs=n_epochs, batch=batch_size)
            self.__save_model()
            self.clear_cache()
            logger.info('Model sucessfully trained')
            return results
        elif model_type == 'UNET':
            pass
        else:
            raise TypeError("no such model")
        logger.info('train method successfully executed')

    def evaluate(self, path_to_dataset: str = None, model_type: str = 'YOLO', use_default_model: bool = False, model_path: str = None):
        logger.info('started evaluate method')
        if model_type == 'YOLO':
            if model_path is not None:
                self.load_model('YOLO', model_path=model_path)
            elif self.model is None:
                self.load_model('YOLO', use_default_model)

            if path_to_dataset is None:
                path_to_dataset = os.environ['YOLO_TRAIN_DATA_PATH']
            elif path_to_dataset != os.environ['YOLO_TRAIN_DATA_PATH']:
                path_to_dataset = self.__get_dataset_yaml_file(path_to_dataset)

            output = self.model.val(data=path_to_dataset, imgsz=int(os.environ['TRAIN_IMAGE_SIZE']))

            # output = {
            #     'mAP50': results.results_dict['metrics/mAP50(M)'],
            #     'precision' : results.results_dict['metrics/precision(B)'],
            #     'recall' : results.results_dict['metrics/recall(B)'],
            #     'f1' : results.box.f1[0]
            # }
            self.clear_cache()
            logger.info('Model sucessfully evaluated')
            return output
        elif model_type == 'UNET':
            pass
        else:
            raise TypeError("no such model")
        logger.info('evaluate method successfully executed')

    def load_model(self, model_type: str = 'YOLO', use_default_model: bool = False, model_path: str = None):
        logger.info('started load_model method')
        if model_type == 'YOLO':
            self.model_type = 'YOLO'
            if model_path is not None:
                self.model = YoloModel(model_path, task=os.environ['TASK_TYPE'])
            elif not use_default_model:
                self.model = YoloModel(os.environ['YOLO_MODEL_PATH'], task=os.environ['TASK_TYPE'])
                logger.info(f"loaded model: {os.environ['YOLO_MODEL_PATH']}")
            else:
                self.model = YoloModel(os.environ['YOLO_PRETRAINED_MODEL_PATH'], task=os.environ['TASK_TYPE'])
                logger.info(f"loaded model: {os.environ['YOLO_PRETRAINED_MODEL_PATH']}")
        elif model_type == 'UNET':
            self.model_type = 'UNET'
            self.model = UnetModel()
            self.model.load(model_path)
            if model_path is not None:
                self.model.load(model_path)
            elif not use_default_model:
                self.model.load(os.environ['UNET_PRETRAINED_MODEL_PATH'])
            else:
                self.model.load(os.environ['UNET_MODEL_PATH'])
        else:
            raise TypeError("no such model")
        logger.info('load_model method successfully executed')

    def __predict_dataset(self, path_to_dataset: str = None, model_type: str = 'YOLO'):
        logger.info('started __predict_dataset method')
        if path_to_dataset is None:
            path_to_dataset = os.environ['PROCESSED_FRAMES_FOLDER']
        if model_type == 'YOLO':
            self.model.predict(source=path_to_dataset, task=os.environ['TASK_TYPE'], save=True, save_txt=True, stream=True) #device='cpu'
        elif model_type == 'UNET':
            pass
        else:
            raise TypeError("no such model")
        logger.info('__predict_dataset method successfully executed')

    def __upscale_predicted_images(self):
        logger.info('started __upscale_predicted_images method')
        image_full_paths = [os.path.join(os.environ['PREDICTED_DATA_PATH'], f) for f in os.listdir(os.environ['PREDICTED_DATA_PATH']) if f.endswith(os.environ['FRAMES_EXTENSION'])]

        for path in image_full_paths:
            input_img = cv2.imread(path, cv2.IMREAD_COLOR)
            resized_image = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (int(os.environ['VIDEO_FRAME_WIDTH']), int(os.environ['VIDEO_FRAME_HEIGHT'])),
                                       interpolation = cv2.INTER_AREA)
            cv2.imwrite(path, resized_image)
        logger.info('__upscale_predicted_images method successfully executed')

    def __calculate_polygon_area(self, xs, ys):
        return 0.5 * np.abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))

    def __handle_file(self, filepath, stream=False):
        logger.info(f'started __handle_file method, file: {filepath}')
        f = open(filepath, "r")
        lines = f.readlines()

        num_of_units = len(lines)

        line_with_max_square_index = -1
        max_square = -1
        best_line_coords = []

        # посчитали
        for i, line in enumerate(lines):
            coords = list(map(float, line.split()[1:]))

            if stream:
                x_coords = np.array(coords[::2]) * int(os.environ['VIDEO_FRAME_WIDTH'])
                y_coords = np.array(coords[1::2]) * int(os.environ['VIDEO_FRAME_HEIGHT'])
            else:
                x_coords = np.array(coords[::2]) * int(os.environ['IMAGE_SIZE'])
                y_coords = np.array(coords[1::2]) * int(os.environ['IMAGE_SIZE'])

            sqaure = self.__calculate_polygon_area(x_coords, y_coords)

            if sqaure > max_square:
                max_square = sqaure
                line_with_max_square_index = i
                best_line_coords = dict({
                    'x' : x_coords,
                    'y' : y_coords
                })

        res = dict({
            'num_of_units' : num_of_units,
            'line_with_max_square_index' : line_with_max_square_index,
            'max_square' : max_square,
            'best_line_coords' : best_line_coords
        })

        logger.info(f'__handle_file method successfully executed')
        return res

    def __handle_labels_dir(self, dirpath):
        logger.info('started __handle_labels_dir method')
        res = []
        onlyfiles = [f for f in listdir(dirpath) if isfile(join(dirpath, f))]

        for filename in onlyfiles:
            fileres = self.__handle_file(os.path.join(dirpath, filename))
            fileres['file_name'] = filename
            image_path = os.path.join(os.environ['PREDICTED_DATA_PATH'], filename.replace(".txt", ".jpg"))
            if os.path.exists(image_path):
                fileres['image_path'] = image_path
            else:
                raise FileNotFoundError('label exists, image is not')
            res.append(fileres)
        logger.info('__handle_labels_dir method successfully executed')
        return res

    def __add_markups(self, markups):
        logger.info(f"started __add_markups method")
        for markup in markups:
            self.__add_markup(markup)
        logger.info('__add_markups method successfully executed')

    def __add_markup(self, markup, stream=False):
        logger.info(f"started __add_markup method {markup['image_path']}")
        dpi = 10
        if stream:
            w = int(os.environ['VIDEO_FRAME_WIDTH']) / (dpi * 10)
            h = int(os.environ['VIDEO_FRAME_HEIGHT']) / (dpi * 10)
        else:
            w = int(os.environ['IMAGE_SIZE']) / (dpi * 10)
            h = int(os.environ['IMAGE_SIZE']) / (dpi * 10)
        data = img.imread(markup['image_path'])
        x = markup['best_line_coords']['x'].astype(int)
        y = markup['best_line_coords']['y'].astype(int)
        fig = plt.figure(frameon=False, figsize=(w, h), dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.text(270, 50,
                f"Est. biggest size: {round(markup['max_square'], 1)} pixels\nNumber of abnormal clods: {markup['num_of_units']}",
                color="white",
                horizontalalignment='center',
                verticalalignment='center')
        ax.plot(x, y, color="white", linewidth=2)
        ax.imshow(data, aspect='auto')
        fig.savefig(markup['image_path'], dpi=dpi * 10)
        logger.info('__add_markups method successfully executed')

    def __proceed_info(self):
        logger.info(f"started __proceed_info method")
        markups = self.__handle_labels_dir(os.path.join(os.environ['PREDICTED_DATA_PATH'], 'labels'))
        self.__add_markups(markups)
        logger.info(f"__proceed_info method successfully executed")

    def predict_video(self, path_to_video: str = 'C:/Users/Acer/Machine Learning/ODaS/clodding_train.mp4', model_type: str = 'YOLO', process: bool = False, predict_on_resized: bool = True, use_default_model: bool = False, model_path: str = None):
        logger.info(f"started predict_video method")
        # model = ODaSModel(model_type)
        self.__split_video_into_frames(path_to_video)
        self.__change_frames(process=process, resize=predict_on_resized)
        self.load_model(model_type, use_default_model, model_path)
        self.__predict_dataset()
        markups = self.__handle_labels_dir(os.path.join(os.environ['PREDICTED_DATA_PATH'], 'labels'))
        self.__add_markups(markups)
        self.__upscale_predicted_images()
        self.__convert_frames_into_video()
        self.clear_cache()
        logger.info(f"predict_video method successfully executed")

    def realtime_detection(self, path_to_video: str, process: bool = True, model_type: str = 'YOLO', use_default_model: bool = False, model_path: str = None):
        logger.info(f"started realtime_detection method")
        if path_to_video is None:
            path_to_video = os.environ['PATH_TO_VIDEO']

        capture = self.__get_capture(path_to_video)

        if model_path is not None:
            self.load_model(model_type, model_path=model_path)
        elif self.model is None:
            self.load_model(model_type, use_default_model, model_path)

        while capture.isOpened():
            gc.collect()
            torch.cuda.empty_cache()
            ret, frame = capture.read()
            if ret:
                if process:
                    frame = self.__apply_brightness_contrast_sharpening(frame, int(os.environ['CONVERT_BRIGHTNESS']) + 255,
                                                                      int(os.environ['CONVERT_CONTRAST']) + 127,
                                                                      int(os.environ['CONVERT_SHARPENING_CYCLE']),
                                                                      prod=True,
                                                                      process=process, resize=False)
                cv2.imwrite(os.environ['TEMP_IMAGE_FILE'], frame)
                self.__predict_dataset(os.environ['TEMP_IMAGE_FILE'], model_type)
                print('temp im fil', os.environ['TEMP_LABEL_FILE'])
                if os.path.exists(os.environ['TEMP_LABEL_FILE']):
                    print('exist')
                    markup = self.__handle_file(os.environ['TEMP_LABEL_FILE'], stream=True)
                    markup['image_path'] = os.environ['PREDICTED_IMAGE_PATH']
                    os.remove(os.environ['TEMP_LABEL_FILE'])
                    self.__add_markup(markup, True)

                image = cv2.imread(os.environ['PREDICTED_IMAGE_PATH'], cv2.IMREAD_COLOR)
                cv2.imshow('Frame', image)
                # add waitKey for video to display
                cv2.waitKey(1)
                if cv2.waitKey(25) == ord('q'):
                    # do not close window, you want to show the frame
                    # cv2.destroyAllWindows();
                    cv2.destroyAllWindows()
                    self.clear_cache()
                    logger.info(f"realtime_detection method finished by user")
                    break
            else:
                logger.info(f"realtime_detection detected all images")
                break


class CliWrapper(object):
    def __init__(self):
        self.segmentator = ODaSModel()
        self.segmentator.setup_env()
        logger.info(f"CliWrapper object inited")

    def train(self, dataset: str = None, model_type: str = 'YOLO', batch_size: int = 4, n_epochs: int = 10):
        self.segmentator.train(dataset, model_type, batch_size, n_epochs)
        print(f"Model saved to {os.environ['YOLO_CUSTOM_TRAIN_MODELS_FOLDER_PATH']}")
        logger.info(f"Model saved to {os.environ['YOLO_CUSTOM_TRAIN_MODELS_FOLDER_PATH']}")

    def evaluate(self, dataset: str = None, model_type: str = 'YOLO', use_default_model: bool = False, model_path: str = None):
        res = self.segmentator.evaluate(dataset, model_type, use_default_model, model_path)
        print(f"Model evaluated, results: {res}")
        logger.info(f"Model evaluated, results: {res}")

    def convert(self, video: str = None, model_type: str = 'YOLO', process: bool = False, predict_on_resized: bool = True, use_default_model: bool = False, model_path: str = None):
        self.segmentator.predict_video(video, model_type, process, predict_on_resized, use_default_model, model_path)
        print(f"Video saved to {os.environ['OUTPUT_FOLDER']}")
        logger.info(f"Video saved to {os.environ['OUTPUT_FOLDER']}")

    def demo(self, video: str = None, model_type: str = 'YOLO', process: bool = True, use_default_model: bool = False, model_path: str = None):
        self.segmentator.realtime_detection(video, process, model_type, use_default_model, model_path)
        self.segmentator.clear_cache()
        logger.info(f"Demo finished")

    def load(self, model_type: str = 'YOLO', model_path: str = None, use_default_model: bool = False):
        self.segmentator.setup_env()
        self.segmentator.load_model(model_type=model_type, use_default_model=use_default_model, model_path=model_path)
        self.segmentator.clear_cache()
        print(f"Successfully loaded {model_type} model")
        logger.info(f"Successfully loaded {model_type} model")


if __name__ == "__main__":
    """Method that run Google's python-fire on CliWrapper class
    to start support of CLI commands"""

    fire.Fire(CliWrapper)
