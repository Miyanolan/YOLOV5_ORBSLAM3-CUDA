# **YOLOV5_ORBSLAM3 (CUDA Version)**  

🚀 **CUDA-Accelerated Version of YOLO_ORB_SLAM3**  

🔗 **Based on:**  
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)  
- [YOLOv5](https://github.com/ultralytics/yolov5)  

Forked from: [YWL0720/YOLO_ORB_SLAM3](https://github.com/YWL0720/YOLO_ORB_SLAM3)  

---

## **📌 Features**
- **Integrated YOLOv5** for object detection within the **ORB-SLAM3** framework.  
- **CUDA 12.1 acceleration** for high-performance inference.  
- **Libtorch 2.4** is used for deep learning inference.  

---

## **📋 Requirements**
Ensure that the following dependencies are installed:  

| Dependency  | Version |
|------------|---------|
| ORB-SLAM3  | ✅ Installed |
| Libtorch   | 2.4 (Located in `Thirdparty/libtorch`) |
| CUDA       | 12.1 |
| OpenCV     | 4.2 |

---

## **🚀 Installation & Build**
Follow these steps to build and run **YOLOV5_ORBSLAM3_CUDA**:  

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/Miyanolan/YOLOV5_ORBSLAM3-CUDA.git
cd YOLOV5_ORBSLAM3-CUDA
```

### **2️⃣ Build the Project**
```bash
chmod +x build.sh
./build.sh
```

---

## **▶️ Running the System**
To run **RGB-D SLAM** with the **TUM dataset**, use the following command:  

```bash
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt \
    ./Examples/RGB-D/TUM3.yaml \
    ./rgbd_dataset_freiburg3_walking_xyz \
    rgbd_dataset_freiburg3_walking_xyz/associations.txt
```

---

## **📜 License**
This project is released under the **MIT License**.  

---

## **📩 Contact**
If you have any questions or issues, feel free to create an issue or reach out! 😊

![ORB3YOLO5](https://github.com/user-attachments/assets/9eef8a6a-3a81-4977-b46e-c28c27883d14)
