# 🌾 Olympic AI 2026 - Phân Loại Bệnh Lúa (Rice Disease Classification)

## 📋 Mục Lục
1. [Giới Thiệu](#giới-thiệu)
2. [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
3. [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
4. [Hướng Dẫn Chạy Lại (Reproduce Instructions)](#hướng-dẫn-chạy-lại)
5. [Chi Tiết Từng Bước](#chi-tiết-từng-bước)
6. [Các Thể Loại Bệnh](#các-thể-loại-bệnh)
7. [Kết Quả](#kết-quả)

---

## 📖 Giới Thiệu

Đây là dự án phân loại bệnh trên lá lúa sử dụng **EfficientNet-B2** và các kỹ thuật augmentation tiên tiến. 

**Mục tiêu:** Xây dựng mô hình AI để phát hiện 8 loại bệnh trên lá lúa từ hình ảnh.

**Model:** EfficientNet-B2 (từ TIMM)  
**Metric chính:** F1-macro Score  
**Framework:** PyTorch + TIMM  

---

## 🗂️ Cấu Trúc Dự Án

```
Olympic-AI-2026-TTV/
├── olympic-ai-2026-ttv-speedup-v5.ipynb    # Notebook chính
├── submission.csv                          # File submission cho Kaggle
├── README.md                               # File hướng dẫn này
└── models/
    ├── best_efficientnet_rice_f1.pth       # Model đã train (best weight)
    └── link_file_weights.txt               # File link download weights
```

---

## 💻 Yêu Cầu Hệ Thống

### Phần Cứng
- **GPU:** NVIDIA GPU (CUDA compatible) - Khuyến khích
- **RAM:** Tối thiểu 8GB (16GB khuyến khích)
- **Dung lượng:** ~5GB cho data + model

### Phần Mềm
```
Python >= 3.9
PyTorch >= 2.0.0 (với CUDA support)
torchvision >= 0.15.0
timm >= 0.9.0
pandas
scikit-learn
numpy
matplotlib
seaborn
Pillow
```

---

## 🚀 Hướng Dẫn Chạy Lại (Reproduce Instructions)

### Tùy Chọn 1: Chạy Trên Kaggle Notebook ⭐ **KHUYẾN KHÍCH**

Đây là cách **tương thích tốt nhất** vì notebook đã được optimize cho Kaggle.

#### Bước 1: Chuẩn Bị
1. Truy cập **[Kaggle Competitions - Olympic AI 2026](https://www.kaggle.com/competitions/fptu-can-tho-olympic-ai-2026)**
2. Vào tab **Notebooks** → **+ New Notebook**
3. Chọn **Python** (hoặc import notebook nếu có)

#### Bước 2: Copy Code
- Copy toàn bộ nội dung từ `olympic-ai-2026-ttv-speedup-v5.ipynb`
- Paste vào Kaggle Notebook mới

#### Bước 3: Chạy Lần Lượt

**Cell 1️⃣ : Chuẩn Bị Dataset**
```python
# Chạy cell này để split train/val
# Input: /kaggle/input/competitions/fptu-can-tho-olympic-ai-2026/train
# Output: /kaggle/working/dataset_split
```
- ✅ Kết quả: Thư mục `train/` và `val/`

**Cell 2️⃣ : Hiển Thị Sample Images**
```python
# Preview 8 hình ảnh mẫu từ các class khác nhau
```

**Cell 3️⃣ : Custom Augmentation Classes**
```python
# Define các augmentation custom:
# - RandomShadow
# - RandomJPEGCompression
# - RandomMotionBlur
# - RandomMoirePattern
# - RandomChromaticAberration
# - v.v.
```

**Cell 4️⃣ : Setup Model & Training**
```python
# Tạo DataLoader, Model EfficientNet-B2, Loss, Optimizer
# ⚠️ NẾU CÓ BEST MODEL CÓ SẴN:
#    - Để model tải lại sau (Cell 7)
#    - HOẶC chạy training từ đầu (mất ~2-4 giờ)
```

**Cell 5️⃣ : Visualize Augmentation**
```python
# Hiển thị 8 phiên bản augmented của 1 ảnh
```

**Cell 6️⃣ : Model Summary**
```python
# In thông tin model + class weights
```

**Cell 7️⃣ : TRAINING 🔥**
```
- Chạy cell này
- Kết quả: best_efficientnet_rice_f1.pth
```

**Cell 8️⃣ : Hiển Thị Kết Quả Training (Plot)**
```python
# Vẽ 3 biểu đồ:
# 1. Loss (Training vs Validation)
# 2. Accuracy (Training vs Validation)
# 3. F1-macro Score
```

**Cell 9️⃣ : Confusion Matrix + ROC Curves**
```python
# - Confusion Matrix
# - ROC Curves (Multi-class)
# - Final Metrics (F1, Accuracy, ROC AUC)
```

**Cell 🔟 : INFERENCE (Dự Đoán Test Set)**
```python
# Tải model best → Dự đoán test set
# Input: /kaggle/input/competitions/fptu-can-tho-olympic-ai-2026/test_speedup
# Output: /kaggle/working/my_submission.csv
```
- ✅ Kết Quả: `my_submission.csv`

**Cell 1️⃣1️⃣ : Phân Tích Kết Quả Prediction**
```python
# Vẽ biểu đồ phân bố prediction trên test set
# In thống kê số lượng hình ảnh theo class
```

#### Bước 4: Submit Kết Quả
1. Sau khi cell 10 chạy xong → File `my_submission.csv` được tạo
2. Vào tab **Notebooks** → **Session** → **Output Files**
3. Download file `my_submission.csv`
4. Quay lại trang competition
5. Vào tab **Submit Predictions**
6. Upload file `my_submission.csv`
7. ✅ Hoàn thành!

---

## 📊 Chi Tiết Từng Bước

### 1️⃣ Chuẩn Bị Dữ Liệu (Cell 1)

```python
# Cấu hình
input_dir = "/kaggle/input/competitions/fptu-can-tho-olympic-ai-2026/train"
output_dir = "/kaggle/working/dataset_split"
split_ratio = 0.8  # 80% train, 20% validation
```

**Kết quả:**
```
dataset_split/
├── train/
│   ├── Bacterial Leaf Blight/    
│   ├── Brown Spot/              
│   ├── Healthy Rice Leaf/       
│   └── ... (8 class tổng cộng)
└── val/
    └── (cùng cấu trúc)
```

---

### 2️⃣ Augmentation Strategy (Custom)

**Các kỹ thuật được sử dụng:**

| Kỹ Thuật | Xác Suất | Mục Đích |
|---------|----------|---------|
| RandomShadow | 30% | Tăng phong phú các điều kiện ánh sáng |
| RandomJPEGCompression | 30% | Giả lập nén ảnh không hoàn hảo |
| RandomMotionBlur | 20% | Xử lý ảnh bị mờ do chuyển động |
| RandomMoirePattern | 35% | Giả lập lỗi quét camera |
| RandomChromaticAberration | 25% | Giả lập sai sắc thái |
| RandomStrongMotionBlur | 25% | Thêm blur mạnh hơn |
| RandomColorShiftAndSharpen | 30% | Biến động màu sắc |
| RandomGridArtifact | 20% | Giả lập lỗi lưới camera |
| RandomErasing | 30% | Occlusion data augmentation |
| Random Noise | N/A | Thêm nhiễu Gaussian |

**Lợi ích:**
- ✅ Giúp model mạnh mẽ với ảnh chất lượng kém
- ✅ Tăng regularization → Giảm overfitting
- ✅ Cải thiện F1 Score ~5-10%

---

### 3️⃣ Model Architecture

**Model:** EfficientNet-B2 (pre-trained)

```python
model = timm.create_model(
    "efficientnet_b2",
    pretrained=True,           # Pre-trained trên ImageNet
    num_classes=8,             # 8 loại bệnh
    drop_rate=0.35,           # Dropout
    drop_path_rate=0.25       # DropPath (Stochastic Depth)
)
```

**Thông Số Model:**
- Tổng parameters: ~7.71 M
- Trainable parameters: ~7.71 M
- Input size: 288×288 pixels
- Output: 8 classes (softmax)

---

### 4️⃣ Training Configuration

```python
# Hyperparameters
batch_size = 8
epochs = 150
learning_rate = 5e-4
weight_decay = 5e-5
img_size = 288
patience = 30  # Early stopping

# Loss Function
criterion = CrossEntropyLoss(
    weight=class_weights,      # Cân bằng class imbalance
    label_smoothing=0.15       # Regularization
)

# Optimizer
optimizer = AdamW(lr=5e-4, weight_decay=5e-5)

# Scheduler
scheduler = CosineAnnealingLR(T_max=150, eta_min=1e-6)
```

**Early Stopping:**
- Tính toán F1-macro trên validation set
- Nếu F1 không cải thiện trong 30 epoch → Dừng training
- Lưu best model với F1 cao nhất

---

### 5️⃣ Inference & Submission

**Quy Trình:**

1. **Load Best Model**
   ```python
   model.load_state_dict(torch.load(best_model_path))
   model.eval()
   ```

2. **Dự Đoán từng ảnh test**
   ```python
   for img_name in test_images:
       image = Image.open(img_path)
       image = transform(image)  # Normalize, Resize
       output = model(image)
       prediction = argmax(output)
   ```

3. **Lưu CSV**
   ```
   image_id, label
   test_001.jpg, 2
   test_002.jpg, 5
   ...
   ```

4. **Format Submission**
   - Column 1: `image_id` (tên file ảnh)
   - Column 2: `label` (số từ 0-7 tương ứng class)

---

## 🌾 Các Thể Loại Bệnh

Mô hình phân loại 8 loại bệnh trên lá lúa:

| # | Tên Bệnh | Mã | Chi Tiết |
|---|----------|-----|---------|
| 0️⃣ | **Bacterial Leaf Blight** | 0 | Bệnh vàng lá do vi khuẩn |
| 1️⃣ | **Brown Spot** | 1 | Đốm nâu trên lá |
| 2️⃣ | **Healthy Rice Leaf** | 2 | Lá lúa khỏe mạnh (không bệnh) |
| 3️⃣ | **Leaf Blast** | 3 | Bệnh lở lá (rất nguy hiểm) |
| 4️⃣ | **Leaf Scald** | 4 | Bệnh cháy lá |
| 5️⃣ | **Narrow Brown Leaf Spot** | 5 | Đốm nâu hẹp |
| 6️⃣ | **Rice Hispa** | 6 | Sâu xanh hại lúa |
| 7️⃣ | **Sheath Blight** | 7 | Bệnh bạc lá (trên vỏ) |

---

## 📈 Kết Quả

### Metric Cuối Cùng (Trên Validation Set)

```
✅ F1-macro Score: 0.9512
✅ Accuracy: 0.9583
✅ ROC AUC: 0.9920
```

### Confusion Matrix
- Model phân loại chính xác nhất với class "Healthy Rice Leaf"
- Một số class dễ bị nhầm với nhau (ví dụ: Brown Spot ↔ Narrow Brown Leaf Spot)

### ROC Curves
- Tất cả class đều có AUC > 0.95
- Model không có overfitting đáng kể

---

## 📚 Tài Liệu Tham Khảo

- **Kaggle Competition:** [Olympic AI 2026](https://www.kaggle.com/competitions/fptu-can-tho-olympic-ai-2026)

---
## 🎉 Hoàn Thành!

**Happy Coding!** 💻

---