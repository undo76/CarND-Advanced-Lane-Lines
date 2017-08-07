## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[distortion]: ./output_images/distortion.jpg
[distortion-2]: ./output_images/distortion-2.jpg
[warped]: ./output_images/warped.jpg
[pipeline]: ./output_images/pipeline.jpg

[distortion]: ./output_images/distortion.jpg
[distortion]: ./output_images/distortion.jpg

[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

All the code can be found in the IPython notebook located in `Advanced Lane Lines.ipynb`.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

In order to calibrate the camera I have to find the corners of the chessboard patterns. OpenCV needs to know the number of (inner) rows and columns. By simple inspection, I find that they are 9 and 6 respectively. There are 3 images where not all the corners are visible. For simplicity I can just skip them.

``` python
NX, NY = 9, 6

def find_chessboards(calibration_imgs, nx, ny):
    ret = []
    for i, img in enumerate(calibration_imgs):
        found, corners = cv2.findChessboardCorners(img, (nx, ny))
        cv2.drawChessboardCorners(img, (nx, ny), corners, found)
        if not found:
            print("WARN: Chessboard not found for image {}. Skipping.".format(i))
        else:
            ret.append(corners)
    return np.stack(ret)
```

OpenCV needs a mapping between the distorted points and the object points, in order to correct the camera distortion.

``` python
def undistort(objpoints, imgpoints, shape=(720, 1280)):
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = \
        cv2.calibrateCamera(objpoints, imgpoints, tuple(reversed(shape)), None, None)
    def apply(x):
        return cv2.undistort(x, cameraMatrix, distCoeffs) 
    return apply
 
def create_objpoints(nx, ny):
    return np.array(
        [[[x, y, 0.] for y in range(6) for x in range(9)]] * len(imgpoints),
        dtype=np.float32)
```
Applying these functions to the chessboards patterns, I obtain this result:

![][distortion]

### Pipeline (single images)

#### 1. Correct the distortion

Using the parameters calculated in the previous step I correct the distortion of the car camera.

![][distortion-2]

#### 2. Apply the pipeline

In order to find the points in the lane lines I tried several combinations of color-spaces, channels, sobel filters and thresholds. In order to facilitate this task, I used a functional style, that allows me to combine and compose the different elements of the pipeline easily. (Check the code for details).

This is the final pipeline that I applied to the project:

``` python
pipeline = compose(
    convert_color(cv2.COLOR_RGB2HLS),
    
    undistort(objpoints, imgpoints, shape=frame.shape[:-1]),
    lambda x: cv2.warpPerspective(x, perspective_matrix, (1280, 720), flags=cv2.INTER_LINEAR),
    
    # It averages both paths of the fork in a final image
    fork(
        compose(
            lambda x: x[..., 1], # Channel L  
            sobel(1, 0, ksize=5),
            scale(1.),
            threshold(0.1, 1., value=.8)
        ),
        compose(
            lambda x: x[..., 2], # Channel S  
            threshold(100, 255, value=1.),
        )
    ),
    
    threshold(0.1, 1., 255),
    lambda x: x.astype(np.uint8),

    convert_color(cv2.COLOR_GRAY2RGB),
)

transformed = pipeline(undistorted)
```

The result of applying this pipeline to the road image is:

![][pipeline]




![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

![][warped]
The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
