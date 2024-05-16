# Multi-task Face Recognition System

## Introduction
This repository contains the implementation code for the paper titled "A single-model multi-task method for face recognition and face attribute recognition in internet of things and visual computing." The paper presents a novel approach that leverages a single deep learning model to perform multiple tasks—face detection, face recognition, and face attribute recognition—simultaneously.

## Abstract from the Paper
The paper introduces a single-model multi-task methodology that efficiently handles face detection, recognition, and attribute recognition using one unified model. This approach significantly boosts inference speed and reduces resource consumption. In dense face environments, it achieves a reduction of 96% in inference time, 49.5% in memory usage, and 59.7% in CPU time. Additionally, the method simplifies the training steps and enhances the model's generalization capabilities across different scenarios.

## System Overview
The implemented system uses a deep learning model, stored as `multi_task_face_model.h5`, to process visual data and extract relevant facial information. This model is designed to be lightweight yet powerful enough to handle complex visual computing tasks in resource-constrained environments such as those typical in the Internet of Things (IoT) applications.

## Repository Files
- `README.md`: This file, providing an overview of the project and paper.
- `blazeface.py`: Script for face detection.
- `celebAload.py`: Utility for loading the CelebA dataset.
- `faceattr.py`: Module for face attribute recognition.
- `facesql.py`: Handling database operations for face data.
- `flow.py`, `newflow.py`: Workflow scripts for processing data through the model.
- `ft.py`: Fine-tuning utilities for the model.
- `loadfer.py`: Script for loading the FER (Facial Expression Recognition) dataset.
- `train_multi_attr.py`: Training script for multi-attribute recognition.
- `multi_task_face_model.h5`: Pre-trained model file.

## Benefits of the Proposed Method
The integration of multiple face-related tasks into a single model not only simplifies the system architecture but also enhances operational efficiency, making it ideal for real-time applications in IoT and similar domains.

For detailed insights into the methodology, performance evaluations, and more, readers are encouraged to refer to the full paper.
