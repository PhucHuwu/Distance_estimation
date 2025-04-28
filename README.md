# Nonlinear Regression: Estimating Hand-to-Camera Distance

This project implements a quadratic nonlinear regression model to estimate the real-world distance between a hand and a camera based on the relative positions of hand landmarks in 2D images. The approach takes into account the nonlinear perspective effect observed in 2D image data when a hand moves closer to or farther from the camera.

## Background

Nonlinear regression is a statistical analysis method used to model the relationship between a dependent variable `y` and one or more independent variables `x`, where the relationship is expressed using nonlinear functions. Common nonlinear functions include powers, logarithms, and exponentials. 

The general form of a nonlinear regression model can be represented as:

```
y = f(x, θ) + ε
```

Where:
- `f(x, θ)`: Nonlinear function of the independent variable(s) `x`, parameterized by coefficients `θ`.
- `ε`: Random noise or error.

In this project, we employ a quadratic model of the form:

```
y = a * x^2 + b * x + c + ε
```

## Problem Statement

The project aims to estimate the real-world distance between a hand and a camera based on 2D image data. Specifically, we:
- Analyzed how the distance between hand landmarks varies with the hand’s position relative to the camera.
- Used the quadratic nonlinear regression model to predict the actual distance from the 2D pixel distance between specific hand landmarks.

### Key Landmarks and Methodology
1. **Landmarks Considered:**
   - **Horizontal axis:** Landmarks 5 and 17.
   - **Vertical axis:** Landmarks 9 and 0.

2. **Rationale:**
   - The distances between these landmarks remain stable during hand movements like grasping or spreading.
   - Adding vertical axis distances reduces errors caused by hand rotation.

3. **Pixel Distance Calculation:**
   - Using the Euclidean formula:
     
     ```
     d = sqrt((x2 - x1)^2 + (y2 - y1)^2)
     ```

4. **Perspective Effect:**
   - Pixel distances increase as the hand moves closer to the camera and decrease as it moves away.

## Quadratic Model Fitting

The quadratic regression model was fitted using pixel distances as input `x` and real-world distances `y` as output:

```
y = a * x^2 + b * x + c
```

The coefficients `a, b, c` were determined using Python’s **numpy.polyfit** function, which minimizes the sum of squared errors between observed and predicted values.

## Data Collection

The collection of experimental data on real-world distances and pixel distances between hand landmarks played a critical role in building an accurate quadratic regression model. The data collection process was conducted systematically to ensure objectivity and reliability. The steps were as follows:

1. **Video Recording:**
   - Data was collected by recording videos under full lighting conditions to ensure sharp image quality and high accuracy in identifying hand landmarks.
   - Specific technical parameters:
     - Resolution: 1280 × 720 pixels
     - Frame rate: 30 fps

2. **Fixed Positions:**
   - The hand was placed at fixed distances from the camera, ranging from 22 cm to 117 cm, with increments of 5 cm.

3. **Pixel Distance Measurement:**
   - The chosen landmarks were points 5-17 and 0-9 on the hand, ensuring high measurement accuracy. Pixel distances were determined by analyzing images captured from the camera, forming relationships between real-world and pixel distances.

4. **Real-World Distance Measurement:**
   - For each recorded pixel distance, the real-world distance from the camera to the hand was precisely measured. Measurements were performed using accurate tools to minimize errors. These real-world distance values were stored and linked to their corresponding pixel distance values, forming a standardized dataset for modeling.

5. **Repetition:**
   - To ensure data reliability, the collection process was repeated multiple times to check data stability and detect any inconsistencies.

The final result of the data collection process was a dataset comprising pixel distances and corresponding real-world distances.

## Key Features
- **Hand Detection:** Utilized MediaPipe Hands for landmark detection.
- **Regression Analysis:** Applied numpy’s polyfit for model fitting.
- **Visualization:** Generated real-time plots of pixel and predicted distances using Matplotlib.
- **Real-Time Processing:** Supported video frame-by-frame hand distance estimation using OpenCV.

## Performance Metrics
- **RAM Usage:** 176.6 MB
- **CPU Usage:** 20.5%
- **Power Consumption:** High

The performance results indicate that the model operates stably under experimental conditions on a medium-configuration computer, suitable for practical applications requiring moderate hardware resources.

## Experimental Conditions
Three videos were tested in different environments to evaluate the model’s applicability under real-world conditions:
- **Person1 - Full Light:** Indoors with uniform lighting.
- **Person2 - Low Light:** Indoors with limited lighting.
- **Person3 - Outdoors:** Natural lighting with varying conditions based on time and camera angle.

## System Configuration
- **Hardware:** Gigabyte G5 GD with Intel i5-11th Gen (6 cores, 12 threads, 4.5GHz), 16GB RAM, NVIDIA GeForce RTX 30 Series.
- **Operating System:** Windows 11 Home (64-bit).
- **Programming Language:** Python 3.10.15.
- **Environment:** Anaconda 9.0.

## Dependencies
- **OpenCV 4.5.5.64:** For video capture and image processing.
- **MediaPipe 0.10.8:** For hand landmark detection.
- **Numpy 1.26.4:** For numerical computations and regression fitting.
- **Matplotlib 3.9.2:** For data visualization.

```bash
pip install opencv-python==4.5.5.64 mediapipe==0.10.8 numpy==1.26.4 matplotlib==3.9.2
```

## Usage
1. Ensure a video file is available at the specified path (default: `video.mp4`).
2. Run the Python script:
   ```bash
   python Estimating_hand_pose.py
   ```
3. Real-time predictions and visualization will be displayed.

## Results
- The quadratic model captures the nonlinear relationship between 2D pixel distances and real-world hand-to-camera distances effectively.
- The approach minimizes errors from perspective effects and provides robust distance estimations for dynamic hand movements.

## Acknowledgments
This project demonstrates the efficacy of quadratic nonlinear regression for practical 3D distance estimation tasks, leveraging the simplicity and robustness of a parabolic model in the context of 2D image data.

