YOLOV5_ORBSLAM3

CUDA VERSION of YOLO_ORB_SLAM3 https://github.com/YWL0720/YOLO_ORB_SLAM3

BASED ON: 1. ORBSLAM3 2.YOLOV5

Requirements : ORBSLAM3, Libtorch 2.4 (in Thridparty/libtorch), CUDA12.1, Opencv4.2

RUN:
cd YOLOV5_ORBSLAM3_CUDA/

. build.sh

./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt ./Examples/RGB-D/TUM3.yaml ./rgbd_dataset_freiburg3_walking_xyz rgbd_dataset_freiburg3_walking_xyz/associations.txt



![ORB3YOLO5](https://github.com/user-attachments/assets/9eef8a6a-3a81-4977-b46e-c28c27883d14)
