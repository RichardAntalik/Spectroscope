import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import sys

display_mode = 1
is_init = 0
cap = None
do_exit = 0

# Calibration, only linear (2 points)
# cal_point = [pixel, wavelength]

def img_to_calibrated(img):
    # Uncomment to calibrate:
    #return img
    
    cal_low = [456, 532] 
    cal_high = [513, 589]
  
    scale_factor = (cal_high[1] - cal_low[1]) / (cal_high[0] - cal_low[0])
    offset = cal_low[1] - (cal_low[0] * scale_factor)
    
    scaled_img = cv2.resize(img, None, fx=scale_factor, fy=1, interpolation=cv2.INTER_CUBIC)
    # could've done affine scale here as well right?...
    M = np.float32([[1,0,offset],[0,1,0]])
    return cv2.warpAffine(scaled_img,M,(img.shape[1],img.shape[0]))

def crop(img, x,y,xx,yy):
    return img[y:yy, x:xx]

def to_gray(img):
    coefficients = [0.5,0.5,0.3]
    # for standard gray conversion, coefficients = [0.114, 0.587, 0.299]
    m = np.array(coefficients).reshape((1,3))
    return cv2.transform(img, m)

#line_img image    
def to_line_img(img):
    maxvalue = 255 * (img.shape[0] / 3)
    line_img = np.sum(img, 0)
    line_img = line_img / maxvalue
    return line_img

def plot_graph(img):
    x,y,xx,yy = plt.axis()
    plt.cla()
    plt.plot(img)
    if (x, y, xx, yy) != (0.0, 1.0, 0.0, 1.0):
        plt.axis([x, y, xx, yy])
        
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')

    #plt.savefig('graph.png')
    plt.pause(0.01)
    
def capture():
    global is_init
    global cap
    if is_init == 0:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_SETTINGS,50)
        is_init = 1
    return cap.read()

def display():
    global display_mode
    ret, frame = capture()
    frame = cv2.flip(frame, 1)
    
    cropped = crop(frame, 400, 350, 1800, 600)
    cropped = img_to_calibrated(cropped)
    
    bw = to_gray(cropped)
    line_img = to_line_img(bw)
    
    if display_mode == 1:
        disp_img = frame
    elif display_mode == 2:
        disp_img = cropped
    elif display_mode == 3:
        disp_img = bw
    elif display_mode == 4:
        # stretch line_img to 2D
        disp_img = np.tile(line_img, (bw.shape[0], 1))
    elif display_mode == 5:
        plot_graph(line_img)
        disp_img = cropped
    else:
        disp_img = frame

    if disp_img is not None:
        cv2.imshow('display', disp_img)
    else:
        cv2.destroyAllWindows()
        
def controls():
    global display_mode
    global do_exit
    key = cv2.waitKey(1) & 0xFF 
    if key == ord('1'):
        display_mode = 1
    if key == ord('2'):
        display_mode = 2
    if key == ord('3'):
        display_mode = 3
    if key == ord('4'):
        display_mode = 4
    if key == ord('5'):
        display_mode = 5
    if key == ord('6'):
        display_mode = 6
    if key == ord('q'):
        do_exit = 1
    
def main():
    global do_exit
    while(True):
        display()
        controls()
        if do_exit == 1:
            break
    cap.release()
    cv2.destroyAllWindows()


main()
