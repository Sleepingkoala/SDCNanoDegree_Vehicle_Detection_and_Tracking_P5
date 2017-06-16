# Udacity Self Driving Car Engineer Nanodegree Project 5

##  Vehicle Detection and Tracking

In this project, my goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), but the main output or product I want to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Creating a great writeup:
---

I will submit my writeup_report as a markdown file, as I usually did earlier.

The Project
---

The goals / steps of this project are the following, as the course mentioned. And I will implement all I studied about traditional computer vision algorithms to see the performances.

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier.
* Optionally, apply a color transform and append spatial binned color features, as well as histograms of color, to combine the HOG feature vector.( Note: for those first two steps don't forget to normalize the features and randomize a selection for training and testing.)
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers(duplicate detections and false positives) and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
* Write a good writeup_report.md.

The Training dataset
---

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of:

- the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html)
- the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/)
- and examples extracted from the project video itself.  

For the project vehicles dataset, the GTI* folders contain time-series data. In the KITTI folder, you may see the same vehicle appear more than once, but typically under significantly different lighting/angle from other instances. You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include them in your writeup for the project by describing what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The Rubric points
---
Here are the [rubric points](https://review.udacity.com/#!/rubrics/513/view) about this project, which I refer to when writing the source code and writeup_report files.

The Optional challenges
---

**As an optional challenge** Once you have a working pipeline for vehicle detection, add in your lane-finding algorithm from the last project to do simultaneous lane-finding and vehicle detection!

**If you're feeling ambitious** (also totally optional though), don't stop there!  We encourage you to go out and take video of your own, and show us how you would implement this project on a new video!
