import cv2

def detect_and_recognize(self):
    self.running = True
    cap = cv2.VideoCapture(self.camera_index)
    
    print(f"攝像頭是否成功開啟: {cap.isOpened()}")
    print(f"攝像頭 FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"攝像頭解析度: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
