# 🛒 Intelligent Shelf Monitoring System Using YOLO-NAS  

Efficient inventory management is the backbone of modern retail. This project showcases a deep learning-based **Intelligent Shelf Monitoring System** designed to revolutionize the way supermarkets handle restocking and product availability.

---

## ✨ Project Overview  

Supermarkets often face challenges in maintaining optimal shelf inventory due to manual monitoring, which is labor-intensive and error-prone. To address this, I developed an advanced **YOLO-NAS model** for real-time shelf monitoring. This system:  

- 🚀 **Reduced restocking time by 25%.**  
- 📈 **Improved product availability tracking** for a seamless customer experience.  
- 🔄 **Automated data integration** with the inventory management system, enabling instant updates.  

With its high-speed and accurate object detection capabilities, the YOLO-NAS model identifies void spaces on shelves, ensuring timely restocking and minimizing stockouts.  

---

## 🔬 Methodology  

1. **Model Selection:**  
   - Utilized **YOLO-NAS**, a next-generation object detection model, for its unparalleled accuracy-speed performance balance.  
   - Fine-tuned the pre-trained model using **PyTorch** and the **SuperGradients library**.  

2. **Dataset:**  
   - Images of supermarket shelves with labeled empty spaces.  
   - Training: 1530 images, Validation: 174 images, Testing: 54 images.  

3. **Key Features:**  
   - **Real-time void detection:** Alerts staff for immediate action.  
   - **Mean Average Precision (mAP):** Achieved an mAP of **0.70-0.85**, ensuring robust performance in real-world scenarios.  

4. **Training Details:**  
   - Cosine annealing learning rate schedule.  
   - Warmup and stage-specific learning rates for efficient convergence.  

---

## 📊 Results  

The YOLO-NAS model demonstrated:  
- Accurate detection of void spaces across various lighting conditions and shelf configurations.  
- A measurable **25% improvement in restocking efficiency** compared to manual methods.  

---

## 🌟 Why This Matters  

Efficient shelf monitoring translates directly to higher sales, improved customer satisfaction, and better resource utilization. This project is a step toward **fully automated retail shelf management**, paving the way for smarter and more efficient stores.  

---

## 📌 Future Work  

- Expanding the system to detect **misplaced items** for better shelf compliance.  
- Incorporating additional classes to track individual product types.  
- Improving performance in low-light or cluttered environments.  

---

## 💡 Technologies Used  

- **YOLO-NAS**: Cutting-edge object detection model.  
- **PyTorch**: Deep learning framework.  
- **SuperGradients**: Model training optimization library.  

---

## 🖼️ Visuals  

Below is a snapshot of the system detecting void spaces in a test image:  

<img width="690" alt="image" src="https://github.com/user-attachments/assets/574b7c24-5136-4df8-a0b2-863d7c7fd385" />


---
