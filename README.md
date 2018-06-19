# ISL Gaze Demo

Demo of appearance-based gaze recognition for the [Intelligent Systems Lab](https://www.ecse.rpi.edu/~cvrl/).

It uses a CNN-based model trained on images of the user's eyes.

## Caveats

Before you attempt to replicate this work, you should be aware that this system
represents a proof-of-concept. It is intended for research applications, and not
for commercial use.

There are several limitations:
- The gaze estimation only works on PCs, and is only tested under Linux.
- The model used is specific to a single person. Therefore, if you want to use
  it on yourself, you will have to collect training data.
- The performance and generalization ability of the model depends on the
  quality of the colleted data.

# Instructions

If you decide that you want to use this code, this section contains instructions
on how to collect data, train a model, and use the model from realtime tracking.

## Installation

Before starting, make sure that you have OpenCV 3 installed, as well as
TensorFlow and the Python liblinear bindings. (The ``python-liblinear``` package
in Ubuntu should work.) Also, make sure that you have TK installed.
(```python-tk``` package in Ubuntu.) Finally, this demo uses
[Keras](https://keras.io/), which can be installed via Pip.

## Data Collection

Data collection is fairly easy. There is a script located in the root of the
repository called ```collect_main.py```. You can run this script to collect data
from yourself:

```
./collect_main.py path/to/images 150
```

This script will display a series of dots, using the webcam on your computer to
capture corresponding images of your face. To properly collect data, it is
imperative that you look at the dots and nowhere else during the entire session.
It is fine to move and rotate your head, as long as you continue looking at the
dots.

The first argument to the script specifies where you want to save the captured
images. The second argument is the number of dots that you want it to display
before exiting. (I usually use 150 dots, as this only takes a few minutes to
collect, but still acheives reasonable screen coverage.)

Upon first starting, the script will show a straight video feed from the webcam
for 10 seconds. This is so you can make sure that the camera is configured
correctly and can see your face. If you would like to skip this step, you can do
so by adding the optional ```-s``` flag.

To produce a good model, it is recommended that you collect at least 15,000
images, ideally covering all environments in which you intend to use the system.
For best results, it is also recommended to use a variety of head orientations
and positions.

## Data Preparation

The training phase expects the data to be packaged into TFRecords datasets for
training and testing. If you collected data with the provided script, it is easy
to convert it into this format. Use the ```extract_crops.py``` script to perform
the conversion:

```
./extract_crops.py path/to/images path/to/output screen_width screen_height
```

The first argument is a path to the images collected. The script expects the
images to be organized internally into folders, one for each session. You need
to point it to the *root* directory, and it will automatically locate and
process all the sessions. For example, if you images were saved to
```data/raw_images/session_1``` and ```data/raw_images/session_2```, point it to
```data/raw_images``` in order to process both.

The second argument is a path to the directory where you want the output
written. It can be any existing directory.

The third and fourth arguments are the width and height, in pixels, of the
screen that you collected your data on.

This script has to run landmark detection on every single image collected, so
running it generally takes awhile, especially for large datasets. When it is
done, it should have created two ```.tfrecord``` files in the output directory.
These are your training and testing datasets.

## Training the model.

The model can be trained by using the ```train_eyes.py``` script:

```
./train_eyes.py path/to/training/data path/to/testing/data
```

The first two arguments are simply the paths to the training and testing
```.tfrecord``` files created in the previous step. The script also takes an
optional ```-o``` flag, which allows you to specify the file that will contain
the saved weights at the end of training. (By default, it saves them to
```eye_model.hd5```.)

If there is a supported GPU available, it will be used for this task.

## Testing the model.

An easy way to test the model is using the ```simple_rt_demo.py``` script. This
script brings up a window, (which should be made fullscreen), and displays a dot
at the estimated location of the user's gaze. It can be run as follows:

```
./simple_rt_demo.py path/to/saved/weights
```

The first argument is a path to the weights that were saved to a file during the
training step.

This script will also use the GPU for inference if there is one available.
