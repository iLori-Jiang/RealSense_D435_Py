from Device.RealSense import RealSense
import numpy as np
import cv2
import os
import sys

if __name__ == "__main__":
    _root_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(_root_path)
    os.chdir(_root_path)
    print('work_dir: ', _root_path)

    camera = RealSense('./Config/rs_d435.yaml')

    frame = camera.get_frame()
    color_image = frame['color_image']
    depth_image = frame['depth_image']

    print(depth_image.shape)

    cv2.imshow('color', color_image)
    key = cv2.waitKey(0)

    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        camera.stop()
