<div>
<h2 align="center">🧑‍🌾 Smart Inventory Management System</h2>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" alt="Python"></a>
  <a href="https://www.mongodb.com/"><img src="https://img.shields.io/badge/Database-MongoDB-green?logo=mongodb" alt="MongoDB"></a>
  <a href="https://opencv.org/"><img src="https://img.shields.io/badge/Computer%20Vision-OpenCV-orange?logo=opencv" alt="OpenCV"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/Deep%20Learning-PyTorch-red?logo=pytorch" alt="PyTorch"></a>
  <a href="https://reactjs.org/"><img src="https://img.shields.io/badge/Frontend-React-61DAFB?logo=react&logoColor=black" alt="React"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Frontend-Streamlit-brightgreen?logo=streamlit" alt="Streamlit"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>
<br>
<div align="center">
    <div>
      <a href="https://drive.google.com/file/d/17tcCczXaBVLZVIWOP7DyQyGZpOlBfoZD/view?usp=sharing" target="_blank">
        <img src="https://img.shields.io/badge/Demo_Video-Click_Here-red?style=for-the-badge&logo=youtube&logoColor=white" alt="Watch Demo Video" />
      </a>
    </div>
  </div>
</div>

---

## 📌 Overview  

The **Smart Inventory Management System** is an **AI-powered solution** designed for **farmers and warehouses** to track, analyze, and manage agricultural products in real-time.  

It leverages **YOLO (You Only Look Once) object detection**, **MongoDB for inventory storage**, and **Streamlit for visualization**. Real-time alerts are sent via **Twilio SMS/WhatsApp** when stock is low or products are near expiry.  

## 💡 Features include: 

-   **👁️ Automated Visual Tracking:** Real-time object detection and counting using a live camera feed and YOLOv11.
-   **📱 SMS Notifications:** Instant Twilio-powered alerts when stock levels hit minimum thresholds.
-   **📊 Interactive Dashboard:** Clean React.js UI to view current stock, item status (`OK`/`LOW`), and update history.
-   **⚙️ Flexible Management:** Easily add new items, set initial stock, and define custom thresholds.
-   **✏️ Manual Stock Control:** Allows for manual adjustments and overrides to ensure data accuracy.

---

## 📽️ Visual Preview  

<p align="center">
  <img src="assets/img3.jpeg" alt="Dashboard Preview" width="30%">
  <img src="assets/img1.jpg" alt="Object Detection with Count" width="30%">
  <img src="assets/img2.jpeg" alt="SMS Notification" width="30%">
</p>

<div align="center">
  <strong>Dashboard Interface</strong> • <strong>Real-time Object Detection</strong> • <strong>SMS Alert System</strong>
</div>

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Camera Feed] --> B(OpenCV Video Capture)
    B --> C{YOLOv11 Object Detection}
    C -- Object Crossed Line --> D[Update Database]
    D --> E{Stock < Threshold?}
    E -- Yes --> F[Trigger Twilio SMS API]
    F --> G[Farmer Receives Alert]
    D --> H[React Frontend]
    H -- REST API --> I[Flask Backend]
    I -- CRUD Operations --> J[( Database)]
```

---

## 📑 Why YOLO?  

YOLO (You Only Look Once) is chosen because:  
- ✅ Real-time **object detection** suitable for farms/warehouses  
- ✅ High accuracy for **small + large object tracking**  
- ✅ Efficient for **edge devices (Raspberry Pi/ESP32-CAM)**  

**Reference Papers:**  
- [YOLOv5: Real-Time Object Detection](https://arxiv.org/abs/1506.02640)  
- [Applications of Deep Learning in Agriculture](https://www.sciencedirect.com/science/article/pii/S0168169919300372)  

---

## 📂 Dataset & Sources    
- Augmented using **Roboflow** for better YOLO training  
---

## 🛠️ Tech Stack  

- **Python 3.9+**  
- **YOLOv5/YOLOv8** (for detection)  
- **OpenCV** (image preprocessing & camera feed)  
- **MongoDB** (NoSQL database for inventory)  
- **Streamlit** (UI Dashboard)  
- **Twilio API** (alerts & notifications)  

---

## ⚙️ Installation & Setup  

### 1. Clone Repository  
```bash
git clone https://github.com/your-username/smart-inventory.git
cd smart-inventory
```  

### 2. Setup Virtual Environment & Install Dependencies  
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux  
venv\Scripts\activate    # Windows  

pip install -r requirements.txt
```  

### 3. Start Application  
```bash
streamlit run app.py
npm start
```  

---

## 📊 Results  

- ✅ **92% detection accuracy** on crop dataset  
- ✅ Real-time **inventory tracking** with MongoDB backend  
- ✅ Automatic **alerts & insights** improve efficiency  

---

## 🔮 Future Scope  

- 🚀 Integration with **AR** for farmer training & visualization  
- 🌍 Multilingual chatbot for crop/fertilizer recommendations  
- 📦 AI-powered **supply chain optimization**  
- 🤖 AI Agents for **autonomous decision-making**  

---

## 📜 License  

This project is licensed under the **MIT License** – see the LICENSE file for details.  

---
