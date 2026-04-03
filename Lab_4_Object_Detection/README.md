# Object Detection Evolution from R CNN to YOLO  

**Student Name:** Bhatt Deep Manish
**Student ID:** 202511015
**Assignment:** Object Detection Evolution from R CNN to YOLO  
**Course:** IT549 Deep Learning  

## Objective

The objective of this assignment is to understand the evolution of object detection models from early region proposal based methods to modern single stage detectors. The work includes implementation and analysis of R CNN, Fast R CNN, Faster R CNN, and YOLO, along with core concepts such as Intersection over Union and Non Maximum Suppression.

## Dataset Description

Fruit Images Dataset for object detection  

The dataset contains three object classes:
- Apple  
- Banana  
- Orange  

Annotations are provided in bounding box format and are used for both visualization and training.

## Tasks Performed

## Preparation: Ground Truth Visualization

- Loaded a random image from the dataset  
- Parsed the corresponding annotation file  
- Converted annotations into bounding box coordinates  
- Visualized ground truth bounding boxes using OpenCV and Matplotlib  

## Task 1: Intersection over Union

- Implemented IoU from scratch  
- Used bounding boxes in [x_min, y_min, x_max, y_max] format  
- Evaluated IoU on three cases:
  - High overlap  
  - Partial overlap  
  - No overlap  

## Task 2: Selective Search

- Applied OpenCV Selective Search  
- Generated region proposals  
- Visualized the first 200 bounding boxes  

## Task 3: R CNN Bottleneck

- Loaded pretrained ResNet18  
- Removed final classification layer  
- Cropped top 100 region proposals  
- Resized each crop to 224 x 224  
- Passed each crop independently through the network  
- Recorded execution time  

## Task 4: Fast R CNN with RoI Pooling

- Passed full image through convolutional layers once  
- Generated feature map  
- Applied RoI Pooling on region proposals  
- Compared execution time with Task 3  

## Task 5: Faster R CNN

- Loaded pretrained Faster R CNN model  
- Performed inference on sample image  
- Extracted bounding boxes, labels, and confidence scores  
- Filtered predictions using confidence threshold  
- Visualized final predictions  

## Task 6: Non Maximum Suppression

- Implemented custom NMS algorithm  
- Sorted boxes based on confidence  
- Removed overlapping boxes using IoU threshold  
- Retained most confident detections  

## Task 7: YOLO Fine Tuning

- Used Ultralytics YOLOv8 model  
- Prepared dataset in YOLO format  
- Trained model for required number of epochs  
- Evaluated performance using:
  - mAP at 50  
  - mAP at 50 to 95  
- Ran inference on test images  
- Visualized predictions  

## Comparison

Comparison between different models based on:

- Inference time per image  
- Precision  
- Recall  
- mAP metrics  

## Technologies Used

- Python  
- PyTorch  
- Torchvision  
- OpenCV  
- Ultralytics YOLO  
- NumPy  
- Matplotlib  

## Repository Structure

- main.ipynb  
- README.md  

## How to Run

1. Clone the repository  
2. Install required dependencies  

```bash
pip install numpy pandas torch torchvision opencv-python matplotlib ultralytics
```

3. Update dataset paths in the notebook  
4. Run main.ipynb  

## Key Learnings

- Understanding of object detection pipeline evolution  
- Importance of IoU in evaluation  
- Limitations of region proposal based methods  
- Efficiency improvements in Fast R CNN and Faster R CNN  
- Role of Non Maximum Suppression  
- Advantages of YOLO for real time detection  

## Conclusion

This assignment demonstrates the progression of object detection techniques from computationally expensive region proposal methods to efficient deep learning based approaches. It highlights how architectural improvements significantly reduce computation while maintaining detection accuracy. YOLO represents a major advancement by framing detection as a single stage problem, enabling faster inference and practical deployment.
