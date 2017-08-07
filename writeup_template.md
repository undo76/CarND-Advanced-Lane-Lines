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
[windows]: ./output_images/windows.jpg

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

In order to find the points in the lane lines I tried several combinations of color-spaces, channels, sobel filters and thresholds. To facilitate this task, I used a functional style, that allows me to combine and compose the different elements of the pipeline easily. (Check the code for details).

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


#### 3. Perspective transform

In order to find the parameters for the perspective transform I found a frame with a straight segment of road. Manually, I found their coordinates and projected them in the warped image, in order to make them parallel. I had to be careful to project the center of the car to the center of the image (not the center of the road). This is the result of applying the transformation:

![][warped]

And this is the mapping that I used:

```python
# Move the points in the image
src = np.float32([
    [691, 450],
    [1120, 700],
    [270, 700],
    [595, 450],
])

dst = np.float32([    
    [1010, -100],
    [1010, 720],
    [354, 720],
    [354, -100]
])

plt.figure(figsize=(20,10))
plt.imshow(undistorted)
plt.scatter(src[:,0], src[:,1], c='r', marker='.')
plt.show()

perspective_matrix = cv2.getPerspectiveTransform(src, dst)
```

#### 4. Identification of lane-line pixels and fit their positions with a polynomial.

Once I have warped the lines into a plane, I need to find the left and the right lines of the lane. I use a histogram to find the two points with more density in the x direction in the bottom of the image. This is the result:

![][histogram]

With this data, I can calculate several data that will be useful in order to calculate the curvature and the offset from the center.

``` python
def find_histogram_peaks(img, y=400, plot_histogram=False):
    hist = np.sum(img[y:, :] > 0, axis=0)
    middle = hist.shape[0]//2
    left = np.argmax(hist[:middle])
    right = np.argmax(hist[middle:]) + middle
    if plot_histogram:
        plt.figure()
        plt.title('Histogram')
        plt.plot(hist, '0.8') 
        plt.plot(left, hist[left], 'r<', label='Left peak')
        plt.plot(right, hist[right], 'r>', label='Right peak')
        plt.legend()
    return left, right
    
peaks = find_histogram_peaks(transformed[..., 0], plot_histogram=True)
print("Peaks: {}".format(peaks))
print("Width: {} pixels".format(peaks[1] - peaks[0]))
center = frame.shape[1]//2
print("Center: {} pixels".format(center))
print("Offsets: {} / {} pixels".format(center - peaks[0], peaks[1] - center))
```

Output:

```
Peaks: (349, 1004)
Width: 655 pixels
Center: 640 pixels
Offsets: 291 / 364 pixels
```

Once I have the starting points, I use the method of sliding windows to find the remaining points. Then, on the next frames it uses the previous frame windows as starting points. I have done some optimizations to the original method that allows the windows to recover in case that no points are found. I use the median, instead of the mean in order to calculate the offset, as it is less sensitive to outliers (points that don't belong to a line) 

``` python
def update_windows(img, peak, prev_medians=None, n_windows=12, margin=90, max_dx=10, plot_points=False):
    dst = np.zeros_like(img)
    h = img.shape[0] // n_windows
    points = []
    new_medians = []
    median = peak
    for i in range(n_windows):
        if prev_medians is None or prev_medians[i] is None:
            start_x = median            
        else:
            start_x = prev_medians[i]
            
        start = (img.shape[0] - i*h, np.clip(start_x, margin, img.shape[1] - margin))
        pts, median, found = get_points(img, start, margin, n_windows=n_windows)
        
        if found:
            points.append(pts)
            dx = median - start_x
            median = start_x + np.clip(dx, -max_dx, +max_dx)
            new_medians.append(median)
            plot_window(dst, pts, start, h, margin, median, found, plot_points)  
            
        else:
            median = start_x
            new_medians.append(None)
            
    if len(points) > 0:
        points = np.hstack(points)
        y, x = fit_points(points, img.shape[0], deg=2)
        curv = calculate_curvature(points, img.shape[0])
        plot_road_lane(dst, x, y, margin)
    else:
        curv = 0
        
    return new_medians, curv, dst
```

Then, I fit the points found using a 2nd-degree polynomy. I use the fitted points to plot, the road. I repeat this fitting, but using world coordinates, as it is not a lineal transformation. This is the result of applying this steps to a frame:

![][windows]

#### 5. Radius of curvature of the lane and the position of the vehicle with respect to center.

I used the following code to calcuate the radius of curvature:

``` python
def calculate_curvature(points, y_eval, ym_per_pix = 3/100, xm_per_pix = 3.7/655):
    # Fit new polynomials to x,y in world space
    pts_y = points[0] * ym_per_pix;
    pts_x = points[1] * xm_per_pix;
    
    c2, c1, c0 = np.polyfit(pts_y, pts_x, deg=2)
    radius = ((1 + (2*c2*y_eval*ym_per_pix + c1)**2)**1.5) / np.absolute(2*c2)
    
    return radius
```
For the offset, it is just a question of calculating the difference of the center of the road and the center of the image in pixels:

``` python
offset = ((x2[-1] + x1[-1]) / 2 - center) * 3.7/655
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is a frame extracted from the video

![][frame]

---

### Pipeline (video)

Here's a [link to my video result](./project-video-result.mp4)

---

### Discussion

The weakest part of my project is the transformation pipeline. If I had had more time, I would have made it dynamic instead of static in order to change the parameters depending on the environment (i.e. light, pavement state, weather). 


