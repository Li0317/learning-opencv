import numpy as np
import HandTrackingModule as htm
import cv2
import time
import autopy

wCam, hCam = 640, 480
frameR = 110

smooth = 8      #平滑值
clocX, clocY = 0, 0         #当前坐标
plocX, plocY = 0, 0         #之前坐标


cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime = 0
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()


while True:
    # 1.找到手部坐标地址
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist, bbox = detector.findPosition(img)   #手的位置和边界框
    # 2.获取中指和食指的指尖坐标，当只有食指时，处于鼠标移动模式。当食指和中指之间的距离小于某一个值时，鼠标单击
    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]          #食指指尖坐标
        x2, y2 = lmlist[12][1:]         #中指指尖坐标

        # 3.确定哪些手指是向上的
        fingers = detector.fingersUP()
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (0, 255, 255), 2)

        # 4.只有食指时，转变为鼠标移动模式
        if fingers[1] == 1 and fingers[2] == 0:

            # 5.在第四步基础上，进行坐标转换，因为电脑显示器屏幕是3200 * 2000，而视频显示窗口为1024 * 640，需要等比例移动

            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
        # 6.添加平滑值减少抖动和闪烁
            clocX = plocX + (x3 - plocX) / smooth
            clocY = plocY + (y3 - plocY) / smooth
        # 7.移动鼠标完成
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 12, (0, 255, 0), cv2.FILLED)
            plocX, plocY = clocX, clocY
        # 8.食指和中指都是向上时，处于点击模式
        if fingers[1] == 1 and fingers[2] == 1:
            # 9.计算手指之间的距离
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # 10.手指之间距离很短时，单击
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 12, (120, 255, 120), cv2.FILLED)
                autopy.mouse.click()
                time.sleep(0.2)


    # 11.显示帧率
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'Fps:{str(int(fps))}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # 12.显示图象
    cv2.imshow("Image", img)
    cv2.waitKey(1)