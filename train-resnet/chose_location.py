import cv2
import numpy as np
img_re = cv2.imread("bg5.png")
p = (0,0)
while True:
    img = img_re
    img = cv2.circle(img_re, p, 3, ( 0, 0,255), -1)
    cv2.imshow("a",img)
    a = cv2.waitKey(1)
    if a == ord("a"):
        p=p[0]-5,p[1]
    elif a == ord("d"):
        p=p[0]+5,p[1]
    elif a == ord("w"):
        p=p[0],p[1]-5
    elif a == ord("s"):
        p=p[0],p[1]+5
    elif a == ord("q"):
        cv2.destroyAllWindows()
        break
    print(p)
