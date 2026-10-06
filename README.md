# Chest X-ray Multi-Label Classification Project

Dự án phân loại đa nhãn ảnh X-quang ngực sử dụng mạng DenseNet-121. Hệ thống nhận đầu vào là một ảnh X-quang và trả lời 14 câu hỏi có/không (ảnh này có bệnh X không).

## Tổng quan luồng xử lý
- Đầu vào: Một ảnh X-quang (224x224).
- Mạng DenseNet-121 trích xuất ra 14 con số thô (logits z).
- Chia cho T của từng bệnh để hiệu chỉnh.
- Tính xác suất p = sigmoid(z / T).
- So với ngưỡng riêng t của từng bệnh.
- Đầu ra: 14 quyết định có/không, kèm 14 xác suất (mỗi bệnh một xác suất, không cộng lại bằng 1).
- Lưu ý: Phần "chia cho T" và "ngưỡng t" không có sẵn trong mạng, chúng được học sau khi huấn luyện xong trên tập validation (việc của Khối D và E).

## Dữ liệu đi qua 5 khối

- Khối A:
  - Đầu vào: Bộ dữ liệu Kaggle (ảnh), Data_Entry_2017.csv, test_list_NIH.txt
  - Công việc: Đổi nhãn thành 14 cột 0/1, đánh dấu ảnh test, ghi đường dẫn ảnh.
  - Đầu ra: meta.csv (112.120 dòng gồm tên ảnh, bệnh nhân, official, path, 14 nhãn).

- Khối B:
  - Đầu vào: meta.csv
  - Công việc: Chia tập dữ liệu theo bệnh nhân (Patient-wise split).
  - Đầu ra: train.csv (75.860 ảnh), val.csv (10.664 ảnh), test.csv (25.596 ảnh).

- Khối C:
  - Đầu vào: train.csv, val.csv và ảnh.
  - Công việc: Huấn luyện DenseNet-121, chọn epoch tốt nhất theo val AUROC.
  - Đầu ra: best.pt và 4 file (logits_val.npy, labels_val.npy, logits_test.npy, labels_test.npy).

- Khối D:
  - Đầu vào: logits_val, labels_val (và test ở cuối).
  - Công việc: Tìm ngưỡng từng bệnh, tính AUROC, AP, F1, khoảng tin cậy.
  - Đầu ra: Bảng Excel (mỗi bệnh một dòng).

- Khối E:
  - Đầu vào: 4 file logits/labels từ Khối C.
  - Công việc: Học hệ số T cho từng bệnh, tính ECE, Brier, vẽ biểu đồ độ tin cậy.
  - Đầu ra: Bảng Excel và ảnh PNG.
