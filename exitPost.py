# coding: utf-8
import cv2, requests, numpy as np
from datetime import datetime
from bs4 import BeautifulSoup as bs

url = 'http://serene-tundra-42600.herokuapp.com/exit'
data = {'enctype': 'multipart/form-data'}

# 内蔵カメラを起動
cap = cv2.VideoCapture(0)
# ラズパイ？iMac?
# cap = cv2.VideoCapture(0)
img = np.zeros((480,640,3), np.uint8)

# OpenCVに用意されている顔認識するためのxmlファイルのパス
# cascade_path = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
# cascade_path = "/home/myjlab/.pyenv/versions/3.6.0/envs/labolog/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt.xml"
cascade_path = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
eye_cascade_path = "/usr/share/opencv/haarcascades/haarcascade_eye.xml"
# カスケード分類器の特徴量を取得する
cascade = cv2.CascadeClassifier(cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# 顔に表示される枠の色を指定（白色）
color = (255,255,255)
# 顔画像のナンバリング用
count = 0

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

                eyes = eye_cascade.detectMultiScale(dst)
                for (ex, ey, ew, eh) in eyes:
                    # 顔画像を切り出して書き出し
                    path = "cutFaces/raw/face" + str(datetime.now()) + "(" + str(count) + ").jpg"
                    cv2.imwrite(path, dst)
                    count += 1
                    # cv2.rectangle(dst, (ex, ey), (ex + ew, ey + eh), color, 2)

                    print('--------------------------\n誰か出てった！')

            # # post
            # image = open(path, 'rb')
            # files = {'exit_face': ('filename.jpg', image, 'image/jpeg')}
            # r = requests.post(url, files=files, data=data)
            # soup = bs(r.text, 'html.parser')
            # if (soup.select_one('div.flash') is None):
            #     print('気がした\n--------------------------')
            # else:
            #     print(soup.select_one('div.flash').string + '\n--------------------------')


        # 表示
        # cv2.imshow("frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

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
