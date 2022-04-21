import cv2
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self, mode = False, maxHands = 2, modelCom = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelCom = modelCom
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelCom, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       #将图象转换为RGB图象
        self.results = self.hands.process(imgRGB)           #处理图象帧，输出结果
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw = True):
        xList = []         #边界框判定中候选x值
        yList = []         #边界框判定中候选y值
        bbox = []          #边界框数组
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):           #获取的lm（landmark）为小数，是相对比例
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)           #这里需要将小数乘上图象大小，转换为实际坐标
                xList.append(cx)        #边界框x值添加
                yList.append(cy)        #边界框y值添加
                #print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 6, (255, 0, 0), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)        #边界框的最大最小x值
            ymin, ymax = min(yList), max(yList)        #边界框的最大最小y值
            bbox = xmin, ymin, xmax, ymax              #边界框
            if draw:                #边界框绘制，边框+-20是让其看起来更好，不会压线
                cv2.rectangle(img, (bbox[0]-20, bbox[1]-20), (bbox[2]+20, bbox[3]+20), (0, 255, 0), 2)
        return self.lmList, bbox

    def fingersUP(self):
        fingers = []
#大拇指判别
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
#其余四根手指判别
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

#手指之间距离计算
    #这里传入的p1, p2是手指的编号
    def findDistance(self, p1, p2, img, draw = True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0
    wCam, hCam = 1024, 640
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0 :
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'Fps:{str(int(fps))}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()