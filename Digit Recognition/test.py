import cv2
import numpy as np

# training part
samples = np.loadtxt('generalsamples.data', np.float32)
responses = np.loadtxt('generalresponses.data', np.float32)
responses = responses.reshape((responses.size, 1))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

# testing part

im = cv2.imread('110_0076.jpg')
out = np.zeros(im.shape, np.uint8)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 20)
c, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt) > 50:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if h > 28:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = thresh[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (10, 10))
            roismall = roismall.reshape((1, 100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
            string = str(int((results[0][0])))
            cv2.putText(out, string, (x, y + h), 0, 1, (0, 255, 0))
# gray = cv2.resize(gray, (500, 768))
# cv2.imshow('gray', gray)

thresh = cv2.resize(thresh, (1920, 900))
cv2.imshow('th', thresh)
im = cv2.resize(im, (1920, 900))
cv2.imshow('im', im)
out = cv2.resize(out, (1920, 900))
cv2.imshow('out', out)
cv2.waitKey(0)
