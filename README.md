# Real-Time Car License Plate Recognition with YOLOv8l

## Introduction

This project focuses on real-time car license plate recognition using the YOLOv8l model. This sophisticated system utilizes computer vision techniques to identify and process license plates on moving vehicles in real-time. The project not only detects license plates but also associates them with their respective vehicles. It opens up possibilities for applications such as smart parking systems, automated access control, and traffic management.

## Data Source

The video used for training and testing the license plate recognition model can be accessed [here](https://www.youtube.com/watch?v=aMgkf_xslQE&t=6s&ab_channel=ExoticCarspotters).

## Models and Training

The core of this project is the YOLOv8l license plate detector model. The model was trained using the YOLOv8 framework, leveraging a carefully curated dataset available [here](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4). Due to hardware limitations (lack of a powerful GPU), the training was conducted in Google Colab, a cloud-based environment. The best.pt weights, representing the trained model, were subsequently downloaded and integrated into the project.

To ensure accurate license plate recognition, the system employs a `sort` algorithm that assigns a unique identifier to each detected vehicle. The system also utilizes custom functions, such as `get_car` and `read_license_plate`, in conjunction with the EasyOCR library to read and correctly associate the license plate text with the respective vehicle. It's important to note that the license plate detector was trained on a relatively small dataset for a limited number of epochs (50) due to computational constraints. This limitation may affect its performance under certain conditions.

## Results

While the video used for testing could not be uploaded due to performance issues on the my laptop, a screenshot of real-time plate recognition is provided below for reference:

![Screenshot](https://github.com/AbzalAidakhmetov/Real_Time_Car_License_Plate_Recognition_YOLOv8l/assets/99760649/e72bf031-87ad-487a-b5be-9278a2cabd8d)

Additionally, the project includes a real-time car counting feature, the repository for which can be found [here](https://github.com/AbzalAidakhmetov/Project_Car_Counter_Yolo.git). When combined, these components can be utilized to create intelligent parking systems capable of automatically and accurately counting the number of vehicles entering and exiting parking facilities while recording their license plate information.

## Dependencies

The following modules were utilized as dependencies in this project:

- The `sort` module was sourced from [this repository](https://github.com/abewley/sort).
- The `util` module was obtained from [this repository](https://github.com/computervisioneng/object-detection-course).
