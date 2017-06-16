# Udacity Self-Driving Car Engineer Nanodegree Project 5

##  Vehicle Detection and Tracking
---

## Writeup Template

In this project, my goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), and the main output or product I want to create is a detailed writeup of the project. 

### The vehicle Detection and Tracking Project

---

The goals / steps of this project are the following, as the course mentioned. And I will implement all I studied about traditional computer vision algorithms to see the performances. ** Most of the source codes are duplicated or inspired by those functions in [udacity course resources: Vehicle Detection and Tracking](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/fd66c083-4ccb-4fe3-bda1-c29db76f50a0/concepts/ea732214-fbc2-42b5-ad40-97a06a95b8ee). And I will do more in my further work.**

* **Step1**: Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier.
* **Step2**: Apply a color transform and append spatial binned color features, as well as histograms of color, to combine the HOG feature vector.( Note: for those first two steps don't forget to normalize the features and randomize a selection for training and testing.)
* **Step3**: Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* **Step4**: Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4)and follow detected vehicles.
* **Step5**: Estimate a bounding box for vehicles detected.
* **Step6**: Write a good writeup_report.md.

[//]: # (Image References)
[image1]: ./output_images/example_visualization.jpg
[image2]: ./output_images/HOG.jpg
[image3]: ./output_images/test_img_recs.jpg

[image4]: ./output_images/test_img_recs1.jpg
[image5]: ./output_images/test_img_rects2.jpg
[image6]: ./output_images/test_img_rects3.jpg
[image7]: ./output_images/test_img_rects4.jpg
[image8]: ./output_images/test_combined_rects.jpg
[image9]: ./output_images/heatmap.jpg
[image10]: ./output_images/thresholded_heatmap.jpg
[image11]: ./output_images/labeled_heatmap.jpg
[image12]: ./output_images/draw_img.jpg
[image13]: ./output_images/all_test_imgs.jpg
[image14]: ./output_images/last_frame.jpg

[video]: ./project_video.mp4

## [Rubric Points](https://review.udacity.com/#!/rubrics/513/view) 
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how I addressed each one. As in my earlier projects, I will submit this writeup as a markdown file.

---

### Histogram of Oriented Gradients (HOG)

### Step1: Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier.

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

This project starts with loading all the `vehicle` and `non-vehicle` examples provided in course resources in the 1st code cell in P5.ipynb. Here is the visualization of 32 car examples and 32 notcar examples from the training dataset. From the 2nd code cell of P5.ipynb I know there are 8792 car(positive)examples and 8968 notcar(negative)examples. Thus the given training dataset is balanced.

![alt text][image1]

This part includes 'calcute the HOG features' part of the section named "Step 1: Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier." in "P5.ipynb". After loading several uselful packages in the 1st code cell of P5.ipynb, the code for extracting HOG features is shown in the 4th code cell of my attached IPython notebook P5.ipynb, by building a function named `get_hog_features()`. The corresponding code is as follows:

```python
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs features and the hog_image if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features
```
As we can see above, the get_hog_features()calls the `skimage.feature.hog()` to get features or both features and visual hog maps, depending on the corresponding indicators `vis` and `feature_vec`. 

I fetched one example from positive and negative examples, respectively. And here is the result using the B-channel from `RGB` colorspace and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. 

![alt text][image2]

#### Step2(optional): Apply a color transform and append spatial binned color features, as well as histograms of color, to combine the HOG feature vector.( Note: for those first two steps don't forget to normalize the features and randomize a selection for training and testing.)

Then I explored extracted HOG and various colorspace features from car example and notcar example arrays, by building a function named `extract_features()`. The code is shown in the 6th code cell of P5.ipynb. The executing code of extracting RGB/HSV/LUV/HLS/YUV/YCrCb colorspace are:
```python
         if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
```
And the code of extracting HOG features is:

```python
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
```

This function could also call the bin_spatial() and color_hist()to extract concatenated and flatten spatial binning color features and color histogram features and combine them all. Whie I chose the YUV colorspace and HOG features in this project. And experiments have proven they can satisfy the basic vehicle detection and tracking requirements. And for better performance, more features should be combined and tried in this part.

Besides, I did data pre-processing in the 7th code cell of P5.ipynb: extracting the combined features about YUV colorspace and shape from car/notcar example dataset. Then do data normalization, randomization and train/validation split.

The code of data normalization is following. The dataset normalization will be necessary if combining different types of features such as HOG and color_hist with bin_spatial features. However, in this project only the HOG and YUV colorspace are used. So this normalization could be optional.
```sh
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64) 
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
```
And the code of splitting data into randomized training and validation sets is as follows:
```sh
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
```
#### 2. Explain how you settled on your final choice of HOG parameters.

Assuming the colorspace is given as `YUV`. Then I tried various combinations of the 4 HOG parameters on the linear SVM classifier based on its performances(accuracy and speed),including `orientation`,`pixels_per_cell`,`cell_per_block` and `hog_channel`. I will take the accuracy as the first crterion. After lots of trial and error, the `leave-one-out` tuning idea can help to settle the final choice. There is my parameter tuning log:

when colorspace = 'YUV'
- Using: 6 orientations 16 pixels per cell and 2 cells per block,hog_channel = 'ALL', Feature vector length: 648
  - 49.35 Seconds to extract HOG features, 0.66 Seconds to train SVC, Test Accuracy of SVC = 0.9702
- Using: 8 orientations 16 pixels per cell and 2 cells per block,hog_channel = 'ALL',Feature vector length: 864
  - 44.67 Seconds to extract HOG features, 0.66 Seconds to train SVC, Test Accuracy of SVC = 0.9702
- Using: 9 orientations 16 pixels per cell and 2 cells per block,hog_channel = 'ALL',Feature vector length: 972
  - 56.82 Seconds to extract HOG features,0.93 Seconds to train SVC,Test Accuracy of SVC = 0.9825
- Using: 10 orientations 16 pixels per cell and 2 cells per block,hog_channel = 'ALL', Feature vector length: 1080
  - 46.74 Seconds to extract HOG features, 1.11 Seconds to train SVC, Test Accuracy of SVC = 0.9828
- Using: 11 orientations 16 pixels per cell and 2 cells per block,hog_channel = 'ALL', Feature vector length: 1188
  - 60.56 Seconds to extract HOG features, 1.41 Seconds to train SVC, Test Accuracy of SVC = 0.9862
- Using: 12 orientations 16 pixels per cell and 2 cells per block,hog_channel = 'ALL', Feature vector length: 1296
  - 55.96 Seconds to extract HOG features,1.2 Seconds to train SVC,Test Accuracy of SVC = 0.9845
- Using: 11 orientations 8 pixels per cell and 2 cells per block, hog_channel = '0', Feature vector length: 2156
  - 67.79 Seconds to extract HOG features,8.33 Seconds to train SVC, Test Accuracy of SVC = 0.9654
- Using: 11 orientations 12 pixels per cell and 2 cells per block, hog_channel = '0',Feature vector length: 704
  - 154.89 Seconds to extract HOG features,3.02 Seconds to train SVC,Test Accuracy of SVC = 0.9507
- Using: 11 orientations 16 pixels per cell and 2 cells per block,hog_channel = '0',Feature vector length: 396
  - 36.46 Seconds to extract HOG features, 1.43 Seconds to train SVC,Test Accuracy of SVC = 0.9578
- Using: 11 orientations 16 pixels per cell and 1 cells per block,hog_channel = '0',Feature vector length: 176
  - 24.02 Seconds to extract HOG features, 1.14 Seconds to train SVC, Test Accuracy of SVC = 0.9462
- Using: 11 orientations 16 pixels per cell and 3 cells per block,hog_channel = '0',Feature vector length: 396
  - 20.67 Seconds to extract HOG features, 0.42 Seconds to train SVC, Test Accuracy of SVC = 0.9434
- Using: 11 orientations 16 pixels per cell and 4 cells per block, hog_channel = '0',Feature vector length: 176
  - 19.64 Seconds to extract HOG features, 0.31 Seconds to train SVC, Test Accuracy of SVC = 0.9324
- Using: 11 orientations 16 pixels per cell and 2 cells per block,hog_channel = '0',Feature vector length: 396
  - 21.36 Seconds to extract HOG features,0.48 Seconds to train SVC, Test Accuracy of SVC = 0.9493
- Using: 11 orientations 16 pixels per cell and 2 cells per block, hog_channel = '1',Feature vector length: 396
  - 21.39 Seconds to extract HOG features,0.52 Seconds to train SVC, Test Accuracy of SVC = 0.9406
- Using: 11 orientations 16 pixels per cell and 2 cells per block, hog_channel = '2',Feature vector length: 396
  - 21.43 Seconds to extract HOG features,0.48 Seconds to train SVC, Test Accuracy of SVC = 0.9068
- Using: 11 orientations 16 pixels per cell and 2 cells per block, hog_channel = 'ALL', Feature vector length: 1188
  - 49.01 Seconds to extract HOG features,1.21 Seconds to train SVC, Test Accuracy of SVC = 0.9817

Therefore, the final choice of HOG parameters is: orientation = 11, pixels_per_cell = 16, cells_per_block = 2, hog_channel = 'ALL'. This solution is obtained through lots of trails and errors.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code of constructing and training a linear SVM classifier is in 9th code cell of P5 ipynb. There is a code slice of initializing and training a SVM instance with linear kernel and default paramters:
```python
vc = LinearSVC()
# Calculate the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
```
And the calculation of accuracy score is executed by calling `svc.score(X_test, y_test), 4)`. The solution of determining selected HOG features and colorspace features is similar to the process of finding the best configuration of HOG parameters. Here is the paramter tuning log:

When I Using 11 orientations 16 pixels per cell and 2 cells per block, hog_channel = 'ALL', Feature vector length: 1188:
- colorspace = 'RGB'
  - 67.79 Seconds to extract HOG features,1.95 Seconds to train SVC, Test Accuracy of SVC = 0.9662
- colorspace = 'HSV'
  - 167.39 Seconds to extract HOG features,1.06 Seconds to train SVC, Test Accuracy of SVC = 0.9797
- colorspace = 'LUV'
  - 55.18 Seconds to extract HOG features,1.09 Seconds to train SVC, Test Accuracy of SVC = 0.9749
- colorspace = 'HLS'
  - 55.79 Seconds to extract HOG features,1.09 Seconds to train SVC, Test Accuracy of SVC = 0.9778
- colorspace = 'YUV'
  - 55.67 Seconds to extract HOG features, 0.92 Seconds to train SVC, Test Accuracy of SVC = 0.9817
- colorspace = 'YCrCb'
  - 59.1 Seconds to extract HOG features, 0.97 Seconds to train SVC, Test Accuracy of SVC = 0.9803
  
Therefore, given assumed HOG paramters, the colorspace feature should be 'YUV' to get the highest accuracy of 0.9817. And the training of a linear classifier is an iterative `leave-one-out` approach.

---


### Sliding Window Search
### Step3: Implement a sliding-window technique and use the trained classifier to search for vehicles in images.

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

This part includes the section named "Step 3: Implement a sliding-window technique and use the trained classifier to search for vehicles in images." in "P5.ipynb". And the code of using a trained SVM classifier to detect cars is shown in the 10th code cell of P5.ipynb, in the function`find_cars()`,which uses hog sub-sampling and makes predictions. Rather than doing the time-comsuming feature extraction on each window individually, the sliding window method extracts the HOG featuresw for the entire image (or selected partial region only once). Then these full-image features are subsampled to get their overlap of each window, considering the window size and fed into the classifier. 

The `find_cars()`evaluates the predicions of classifier on HOG features for each window and returns a list of bounding boxes, i.e. the corresponding windows which generated positive predictions. The number of bounding boxes is computed in the 11the code cell. And the returned bounding boxes are drawn onto the original image using function `draw_boxes()` in the 12th code cell in P5.ipynb.

Here each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75% in y direction(yet still 50% overlap in x direction). Its possible to run this same function multiple times for different scale values to generate multiple-scaled search windows. There is an example result using sliding window search as follows, where the bounding boxes include true positive and false positive detections.
![alt text][image3]

Then draw all the potential search regions. The size and position of cars in the image will be different depending on their distance from the front-facing camera, thus find_cars() will be called several times with different `ystart`, `ystop`, and `scale` values. These next few blocks of code are for determining the values for these parameters that work best in the 14th- 17th code cell in P5.ipynb. And the code of combining all the scales is in the 18th code cell.

scale = 1.0, 78 bboxes|![alt text][image4]
----------------------|-------------------
scale = 1.5, 50 bboxes|![alt text][image5]
scale = 1.8, 42 bboxes|![alt text][image6]
scale = 3.0, 24 bboxes|![alt text][image7]
combine all scales    |![alt text][image8]

It seems that the larger the scale is, the less windows are searched. And less false positive can be produced. Correspondingly, the start and stop coordinates are important, but not that important as scale factor. Besides, only an appropriate y-direction range of the image is considered useful to reduce the false positive detection where the vehicles at that scale are unlikely to appear.

From the result image after combining all scales, there are still some redundant true positive and false positive detections. Thus the heatmap strategy is applied to reject duplicate detections and false positives. In practice, a true positive happens usually with several positive detections, while false positives are usually companied with only one or two detections. Based on this fact and multiple detection locations, a combined heatmap and thresholding is used to distinct the two, by calling function `add_heat()` in the 19th code cell in P5 ipynb. In heatmaps, the pixel intensities(the degree of heat) out of a whole black image are enhanced where each detection is located. The more overlapping areas are, the higher intensities of heat are assigned. The following is a heatmap of the combined-all-scales images.

![alt text][image9]

Then a threshold is used to reject some low-intensity pixels(detections) to the heatmap in the function `apply_threshold` in the 21 code cell. In this project the heat threshold is 1. And here comes the thresholded heatmap:

![alt text][image10]

From above we can see the areas affected by false positives are rejected.In practice, I will integrate a heatmap over several continuous frames of video, such that areas of multiple detections get "hot", while transient false positives stay "cool". And to figure out how many cars in each frame and which pixels belong to which cars, the `scipy.ndimage.measurements.label()` function collects spatially contiguous areas of the heatmap and assigns each a label. There are two cars found in the image below. The corresponding code is in the 23rd code cell.

![alt text][image11]

Then I can remap these two labeled detections onto the original images, by building a function `draw_labeled_bboxes()`in the 24th code cell. And this is the result image.

![alt text][image12]

---

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

There are six given test images in folder test_images. The following are the results of using my pipeline to all these six images and their heatmaps. And you can see it works well, and detects near-field of the vehicles and no false positive. 

In my experiments, at first the single b-channel and HOG features are not enough to achieve satisfying accuracy. Then I tried various configurations about colorspaces, YUV with proper HOG parameters the SVM classifier can achieve an accuracy of 0.9862. And I chose to combine 4 scales subsampling to reduce redundent detections. Lowering the heatmap threshold is also useful solution to improve the detection accuracy.

![alt text][image13]


---

### Video Implementation
### Step4: Run the pipeline on a video stream and follow detected vehicles.

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

My pipeline starts with the test_video.mp4 and later implements on full project_video.mp4. And the result is saved as project_video_result.mp4.

Here's a [link to my video result](./project_video_result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The function `video_processing` is in the 40 code cell in P5.ipynb for including the pipeline on video.The difference between my pipeline for video and the pipeline for single image is the video pipeline stores the detected bboxes from previous 15 frames using class `Vehicle_Detect()`. Rather than applying heatmaps/thresholded heatmaps/heatmap labelling mentioned above, the detection for the previous 15 frames are combined and added to the heatmap and the threshold is set to 1 + len(det.prev_rects)//2 (one more than half the number of rectangle sets contained in the history) empirically(rather than using a single scalar, or the full number of rectangle sets in the history). This strategy has filtered false positives efficiently.

---

### Here the resulting bounding boxes are drawn onto the last frame in the series:
####  Step5: Estimate a bounding box for vehicles detected.
Here is a snapshot of the resulting bbox in last frame of the series as follows:
![alt text][image14]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In my project_video_result.mp4, you can see these problems where my pipeline is likely to fail:

- there are still some temporal false positive jumping out at one specific frame.
- And the detected windows are not that suitable for each running vehicle. 
- The window changes drastically when two cars have occlusions.

Since I just did the basic traditional computer vision menthods to detect and track vehicles. The algorithm are very easy and I have lots of work to do in the further, which may include:
- more robust and useful fearture combination should be used.
- use unlinear SVM classifier, through it will take longer time to train and evaluate.
- more labeled data should be used, like the dataset provided by udacity with corresponding csv files.
- try other classifiers, and the state-of-the-art: YOLO framework, given in "You Only Look Once: Unified Real-time Object Detection"(CVPR2016).