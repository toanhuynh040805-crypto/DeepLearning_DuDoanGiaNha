# 🏠 Dự Đoán Giá Nhà Bằng Mạng ANN (Artificial Neural Network)

Đề tài nghiên cứu và thực hành xây dựng mô hình Học Sâu (Deep Learning) ứng dụng mạng nơ-ron nhân tạo cấu trúc Feedforward để giải quyết bài toán hồi quy (Regression): Dự đoán giá nhà dựa trên các thuộc tính tự nhiên, địa lý và kinh tế xã hội.

---

## 📌 1. Giới thiệu đề tài
Bài toán dự đoán giá bất động sản là một bài toán kinh điển nhưng luôn có tính thực tế cao. Dự án này ứng dụng **Mạng nơ-ron nhân tạo (ANN)** với các lớp ẩn liên kết đầy đủ (Dense Layers) để tự động học các mối quan hệ phi tuyến phức tạp giữa các đặc trưng đầu vào và giá trị nhà ở trên thực tế.

* **Công nghệ sử dụng:** Python, TensorFlow/Keras, Scikit-Learn, Pandas, Numpy.
* **Giao diện trải nghiệm:** Streamlit Web Application.
* **Tập dữ liệu thử nghiệm:** California Housing Prices (Kaggle Dataset với hơn 20,000 dòng dữ liệu thực tế).

---

## 📊 2. Mô tả dữ liệu & Tiền xử lý
Mô hình sử dụng các đặc trưng (Features) đầu vào đại diện cho cấu trúc, vị trí và kinh tế của khu vực:

### Các đặc trưng đầu vào (Inputs):
* `total_rooms`: Tổng số phòng của căn nhà.
* `housing_median_age`: Tuổi đời trung bình của ngôi nhà (thay cho năm xây dựng).
* `latitude` & `longitude`: Vị trí địa lý (Kinh độ & Vĩ độ).
* `median_income`: Thu nhập bình quân của hộ gia đình trong khu vực.
* `ocean_proximity`: Vị trí tương đối so với biển (được mã hóa).

### Biến mục tiêu (Target Output):
* `median_house_value`: Giá nhà trung bình (USD).

### Quy trình tiền xử lý dữ liệu:
1. **Xử lý giá trị khuyết thiếu (Missing Values):** Điền các giá trị trống trong tập dữ liệu bằng phương pháp trung vị (`Median Imputation`).
2. **Mã hóa dữ liệu phân loại (Categorical Encoding):** Chuyển đổi thuộc tính định danh `ocean_proximity` thành dạng số bằng kỹ thuật `One-Hot Encoding`.
3. **Chuẩn hóa dữ liệu (Feature Scaling):** Áp dụng `StandardScaler` đưa các biến về cùng phân phối chuẩn (Mean = 0, Std = 1) giúp mạng ANN hội tụ nhanh và ổn định hơn.

---

## 🧠 3. Kiến trúc mạng ANN
Mô hình được thiết kế theo cấu trúc **Multilayer Perceptron (MLP)** bao gồm:

* **Lớp đầu vào (Input Layer):** Tiếp nhận các đặc trưng sau khi đã tiền xử lý.
* **Các lớp ẩn (Hidden Layers):** Gồm 4 lớp ẩn tuyến tính (`Dense`) với số lượng nút lần lượt là `128 -> 64 -> 32 -> 16`.
  * Sử dụng hàm kích hoạt **ReLU** (`Rectified Linear Unit`) tại các lớp ẩn để mô hình hóa các hàm phi tuyến.
  * Tích hợp lớp **Dropout (10%)** sau lớp ẩn đầu tiên để giảm thiểu hiện tượng quá khớp (`Overfitting`).
* **Lớp đầu ra (Output Layer):** Gồm `1 nút` duy nhất với hàm kích hoạt tuyến tính (`Linear`) để dự đoán giá trị số liên tục.

* **Thuật toán tối ưu:** `Adam Optimizer` với tốc độ học (Learning rate) tự điều chỉnh.
* **Hàm mất mát (Loss Function):** `Mean Squared Error (MSE)`.

---

## 📈 4. Kết quả đánh giá mô hình
Mô hình được đánh giá độc lập trên tập dữ liệu kiểm tra (Test Set chiếm 20%) thông qua hai chỉ số đo lường cốt lõi:

* **Mean Absolute Error (MAE):** Đo lường độ lệch tuyệt đối trung bình giữa giá nhà dự đoán và giá nhà thực tế.
* **Mean Squared Error (MSE):** Hàm phạt bình phương sai số, dùng để đánh giá mức độ nghiêm trọng của các lỗi dự đoán lớn.

*(Bạn có thể chèn ảnh đồ thị Loss/MAE hoặc bảng kết quả thực tế vào đây sau khi chạy)*

---

## 💻 5. Hướng dẫn cài đặt và Khởi chạy ứng dụng Web

Dự án tích hợp giao diện Web trực quan bằng **Streamlit**, cho phép người dùng nhập thông số căn nhà bất kỳ và nhận kết quả dự đoán ngay lập tức.

### Yêu cầu hệ thống:
* Đã cài đặt Python (Phiên bản từ 3.8 trở lên).

### Các bước khởi chạy:

1. **Cloning dự án hoặc di chuyển vào thư mục chứa code:**
   ```bash
   cd App_Du_Doan_Gia_Nha
