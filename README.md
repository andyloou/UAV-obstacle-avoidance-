# UAVindoor
ỨNG DỤNG MÔ HÌNH RESNET VÀO BÀI TOÁN UAV TRÁNH VẬT CẢN TỰ ĐỘNG  

Ngày nay, máy bay quad-copters có rất nhiều ứng dụng trong thực tế. Ưu điểm của thiết bị quad-copter là khả năng giữ ổn định vị trí bay và di chuyển theo nhiều hướng dựa trên sự điều khiển tốc độ quay của từng cánh quạt. Ngoài ra, loại UAV này còn có thể gắn thêm nhiều cảm biến khác để phục vụ quá trình điều khiển, thu thập thông tin cho nhiều mục đích. Đó là lý do để phát triển nhiều kỹ thuật điều khiển mới cho từng tình huống.Các ứng dụng với UAV trong nhà có thể phát triển mạnh mẽ dựa trên kỹ thuật AI, với hai vấn đề lớn là tìm đường và tránh chướng ngại vật

Thuật toán Resnet: 
ResNet (Residual Network) được giới thiệu đến công chúng vào năm 2015 và thậm chí đã giành được vị trí thứ 1 trong cuộc thi ILSVRC 2015 với tỉ lệ lỗi top 5 chỉ 3.57%. Giải pháp mà ResNet đưa ra là sử dụng skip connection-kết nối "tắt" đồng nhất để xuyên qua một hay nhiều lớp. Một khối như vậy được gọi là một Residual Block, như trong hình sau :
                                                                                                                                          ![resnet](https://user-images.githubusercontent.com/129571170/236088880-3b5cad95-8a67-4c03-b08a-89ec603877bb.png)

Mô hình Resnet8:

![resnet8](https://user-images.githubusercontent.com/129571170/236089104-28e5ea1b-1c68-4bf7-9d0e-a464719d4e41.png)

Mô hình Resnet50:

![resnet50](https://user-images.githubusercontent.com/129571170/236089250-3244d912-61a5-47d8-9969-281fd6c5702a.png)

Biểu đồ thể hiện va chạm của 2 mô hình sau 80 bước:

![Screenshot 2023-05-04 083641](https://user-images.githubusercontent.com/129571170/236090052-424ac1da-94f2-47ab-b443-d4afd5bb4792.png)

Nhìn vào biểu đồ biểu thị va chạm của 2 mô hình Resnet8 và Renset50 được tính trung bình qua 10 lần đo ta thấy Resnet8 va chạm ít hơn 1 nửa so với Resnet50
Nguyên nhân đề xuất:
  
  •	Mô hình Resnet8 xây dựng tối ưu hơn 
  
  •	Mô hình Resnet50 quá phức tạp so với yêu cầu bài toán với số lượng layers lớn hơn nhiều dẫn đến quá trình kết quả training chưa tốt, khi vào bài toán nhận dạng kém chính xác hơn

