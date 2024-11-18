# Mask R-CNN Day/Night Detection

Model phát hiện đối tượng sử dụng Mask R-CNN, tự động nhận diện ảnh ngày/đêm.

## Requirements
- Docker
- NVIDIA GPU với CUDA support
- NVIDIA Container Toolkit

## Cài đặt
1. Cài đặt Docker: [Docker Installation Guide](https://docs.docker.com/get-docker/)
2. Cài đặt NVIDIA Container Toolkit: [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Sử dụng

1. Pull Docker image:
```bash
docker pull chisnguyen/detectron2-daynight:latest
```

2. Chuẩn bị dữ liệu:
- Tạo thư mục chứa ảnh cần predict
- Tạo thư mục để lưu kết quả

3. Chạy inference:

PowerShell:
```powershell
docker run --gpus all `
-v "<đường_dẫn_thư_mục_chứa_ảnh_test>":/app/input `
-v "<đường_dẫn_thư_mục_lưu_kết_quả>":/app/output `
chisnguyen/detectron2-daynight:latest
```

CMD:
```bash
docker run --gpus all ^
-v "<đường_dẫn_thư_mục_chứa_ảnh_test>":/app/input ^
-v "<đường_dẫn_thư_mục_lưu_kết_quả>":/app/output ^
chisnguyen/detectron2-daynight:latest
```

Ví dụ:
```powershell
docker run --gpus all `
-v "C:\Users\username\test_images":/app/input `
-v "C:\Users\username\results":/app/output `
chisnguyen/detectron2-daynight:latest
```

## Output Format
Kết quả sẽ được lưu trong file predictions.txt với format:
```
image_name class_id x_center y_center width height confidence
```

## Lưu ý
- Đường dẫn trong Windows cần đặt trong dấu ngoặc kép nếu có dấu cách
- Cần có NVIDIA GPU và driver phù hợp
- Container cần quyền đọc/ghi vào thư mục được mount

## Support
Nếu gặp vấn đề, vui lòng tạo issue hoặc liên hệ: [your-contact]
