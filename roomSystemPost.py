# coding: utf-8
import cv2, time, requests, numpy as np
from datetime import date
from bs4 import BeautifulSoup as bs

url = 'http://serene-tundra-42600.herokuapp.com/'
data = {'enctype': 'multipart/form-data'}

if __name__ == "__main__":

    # 内蔵カメラを起動
    maccap = cv2.VideoCapture(0)
    webcap = cv2.VideoCapture(1)
    imgL = np.zeros((480,640,3), np.uint8)
    imgR = np.zeros((480,640,3), np.uint8)

    # OpenCVに用意されている顔認識するためのxmlファイルのパス
    cascade_path = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
    # カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier(cascade_path)

    # 顔に表示される枠の色を指定（白色）
    color = (255,255,255)
    # 顔画像のナンバリング用
    count = 0

    while True:

        # 内蔵カメラから読み込んだキャプチャデータを取得
        macret, macframe = maccap.read(imgL)
        webret, webframe = webcap.read(imgR)

        # 顔認識の実行
        macfacerect = cascade.detectMultiScale(macframe, scaleFactor=1.2, minNeighbors=2, minSize=(10,10))
        webfacerect = cascade.detectMultiScale(webframe, scaleFactor=1.2, minNeighbors=2, minSize=(10,10))

        if len(macfacerect) > 0:
            for rect in macfacerect:
                print('--------------------------\n誰か入ってきた！')
                cv2.rectangle(macframe, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), color, thickness=2)
                x = rect[0]
                y = rect[1]
                width = rect[2]
                height = rect[3]
                dst = macframe[y:y+height, x:x+width]

                # 顔画像を切り出して書き出し
                path = "cutFaces/raw/face" + str(date.today()) + "(" + str(count) + ").jpg"
                cv2.imwrite(path, dst)
                count += 1
                # post
                image = open(path, 'rb')
                files = {'enter_face': ('filename.jpg', image, 'image/jpeg')}
                r = requests.post(url + 'enter', files=files, data=data)
                soup = bs(r.text, 'html.parser')
                if (soup.select_one('div.flash') is None):
                    print('気がした\n--------------------------')
                else:
                    print(soup.select_one('div.flash').string + '\n--------------------------')


        if len(webfacerect) > 0:
            for rect in webfacerect:
                print('--------------------------\n誰か出てった！')
                cv2.rectangle(webframe, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), color, thickness=2)
                x = rect[0]
                y = rect[1]
                width = rect[2]
                height = rect[3]
                dst = webframe[y:y+height, x:x+width]

                # 顔画像を切り出して書き出し
                path = "cutFaces/raw/face" + str(date.today()) + "(" + str(count) + ").jpg"
                cv2.imwrite(path, dst)
                count += 1

                # post
                image = open(path, 'rb')
                files = {'exit_face': ('filename.jpg', image, 'image/jpeg')}
                r = requests.post(url + 'exit', files=files, data=data)
                soup = bs(r.text, 'html.parser')
                if (soup.select_one('div.flash') is None):
                    print('気がした\n--------------------------')
                else:
                    print(soup.select_one('div.flash').string + '\n--------------------------')

        # 表示
        cv2.imshow("macframe", macframe)
        cv2.imshow("webframe", webframe)
          
        # qキーを押すとループ終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 内蔵カメラを終了
    cap.release()
    cv2.destroyAllWindows()
