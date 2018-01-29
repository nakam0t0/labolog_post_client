# coding: utf-8
import cv2, requests, numpy as np

# 内蔵カメラを起動
cap = cv2.VideoCapture(0)
# ラズパイ？iMac?
# cap = cv2.VideoCapture(0)
img = np.zeros((480,640,3), np.uint8)

# OpenCVに用意されている顔認識するためのxmlファイルのパス
cascade_path = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
# カスケード分類器の特徴量を取得する
cascade = cv2.CascadeClassifier(cascade_path)

# 顔に表示される枠の色を指定（白色）
color = (255,255,255)

print('start')

try:
    while True:

        # 内蔵カメラから読み込んだキャプチャデータを取得
        ret, frame = cap.read(img)

        # 顔認識の実行
        facerect = cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=2, minSize=(10,10))

        if len(facerect) > 0:
            for rect in facerect:
                cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), color, thickness=2)
                x = rect[0]
                y = rect[1]
                width = rect[2]
                height = rect[3]
                dst = frame[y:y+height, x:x+width]

                print('face!')

        # 表示
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print()
    print('finish')
    # 内蔵カメラを終了
    cap.release()
    cv2.destroyAllWindows()

except KeyboardInterrupt:
    print()
    print('finish')
    # 内蔵カメラを終了
    cap.release()
    cv2.destroyAllWindows()
