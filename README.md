# Nhận diện khuôn mặt từ ảnh nhiều góc và webcam laptop

## Chạy trên Windows

1. Cài Python 3.10–3.12, chọn Tcl/Tk và Python launcher khi cài.
2. Nhấp đúp `START.bat`. Lần đầu chương trình tạo `.venv` và cài thư viện.
3. Nhập tên người, chọn **Nạp ảnh nhiều góc**, rồi chọn các ảnh của đúng người đó.
4. Bấm **Train / Tạo vector database**, đợi xử lý xong.
5. Bấm **Mở camera**. Camera laptop mặc định là `0`; thử `1` nếu dùng webcam khác.

Lần đầu train/mở camera, ứng dụng tải hai model công khai từ OpenCV Zoo, khoảng 37 MB, kiểm tra SHA-256 rồi lưu trong `data/models/`. Sau đó có thể chạy offline. Ảnh và vector được xử lý cục bộ, không gửi tới API bên ngoài. Chương trình chỉ ghi ảnh webcam khi bạn bấm **Chụp ảnh đăng ký**.

## Chuẩn bị dữ liệu train

Mỗi người có một thư mục, tên thư mục là tên cần hiển thị:

```text
train/
  Nguyen Van A/
    chinh_dien.jpg
    nghieng_trai.jpg
    nghieng_phai.jpg
    ...
  Tran Thi B/
    01.jpg
    02.jpg
    ...
```

- Nên có 10–30 ảnh mỗi người: chính diện, quay trái/phải khoảng 15–45 độ, nhìn lên/xuống nhẹ, khoảng cách và ánh sáng khác nhau.
- Ảnh cần rõ, không quá tối/chói và chỉ có đúng một khuôn mặt. Góc nghiêng quá mạnh có thể không đủ điểm đặc trưng để căn chỉnh.
- Cần ít nhất 3 ảnh khác nội dung vượt qua kiểm tra chất lượng mới đăng ký được một người. Ảnh trùng, mờ, mặt quá nhỏ hoặc có nhiều mặt bị bỏ qua và ghi lý do trong nhật ký.
- Dùng ảnh mới, chưa có trong train, để kiểm tra nhận diện. Thử cả người đã đăng ký và người chưa đăng ký.
- Có thể mở camera trước khi có dữ liệu, nhập tên và bấm **Chụp ảnh đăng ký** nhiều lần với các góc khác nhau, rồi tắt camera và bấm Train.
- Bấm Train sẽ cập nhật vector của những người có đủ ảnh hợp lệ. Người thiếu ảnh giữ vector cũ nếu đã đăng ký. Train không tự xóa người không còn trong thư mục.
- **Xóa vector người đã chọn** chỉ xóa đăng ký trong database. Ảnh train còn nguyên; train lại các ảnh đó sẽ đăng ký lại người này.

## Train trong ứng dụng hoạt động thế nào

Đây là đăng ký đặc trưng, không fine-tune mạng từ đầu: YuNet phát hiện khuôn mặt và 5 điểm đặc trưng; SFace căn chỉnh, tạo embedding 128 chiều; ứng dụng chuẩn hóa L2 rồi lưu từng góc trong SQLite (`data/face_vectors.sqlite3`). Khi test, vector webcam được so cosine với các vector đã đăng ký; trung bình 3 góc khớp nhất đại diện cho điểm của một người.

Tên chỉ được chấp nhận nếu điểm cao hơn ngưỡng, đủ chênh với người đứng thứ hai và khớp liên tiếp 3 lần trên cùng vùng mặt theo tracking IoU. Dự đoán yếu/khác tên làm mất xác nhận ngay. Tracking đơn giản có thể nhầm khi nhiều người che nhau hoặc đổi chỗ: dùng một người trước camera để kiểm tra ban đầu. Ứng dụng không có cơ chế chống giả mạo bằng ảnh/video, không dùng như hệ thống mở khóa an ninh.

Hiển thị:

- `face_detected: Nguyen Van A`: đã xác nhận tên, khung xanh.
- `face_detected: unknown`: chưa khớp đủ, khung đỏ; khung vàng là đang chờ đủ 3 lần xác nhận.
- `cos=0.61` là điểm tương đồng cosine, không phải xác suất chính xác 61%.

Ngưỡng mặc định 0.50 và chênh lệch 0.06 là cấu hình ban đầu, không bảo đảm chính xác cho mọi bộ ảnh. Tăng ngưỡng khi người lạ bị nhận nhầm; giảm nhẹ khi người đúng thường bị unknown và đã có ảnh train rõ, đa góc. Chọn ngưỡng bằng ảnh test riêng, không chỉ thử lại ảnh train. Mở lại camera để áp dụng thay đổi.

## Chạy bằng dòng lệnh

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe face.py --check
.\.venv\Scripts\python.exe face.py --download-models
.\.venv\Scripts\python.exe face.py --train
.\.venv\Scripts\python.exe face.py
```

Đổi nơi lưu/nguồn ảnh:

```powershell
python face.py --train-dir "D:/AnhKhuonMat" --data-dir "D:/FaceData" --camera 0
```

Nếu máy offline, tải thủ công đúng hai tệp bên dưới vào `data/models/`. Chương trình kiểm tra checksum trước khi nạp. Nếu model bị lỗi, đổi tên tệp lỗi rồi chạy tải lại. Nếu camera không mở, đóng Zoom/Teams/Camera và kiểm tra Windows Settings > Privacy & security > Camera > Let desktop apps access your camera.

## Nguồn mô hình

- [OpenCV DNN face detection and recognition](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html)
- [YuNet OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- [SFace OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
- YuNet: `https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx`
- SFace: `https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx`

Chưa có dữ liệu train cá nhân đi kèm, nên chưa có số đo độ chính xác trên người dùng của bạn.
