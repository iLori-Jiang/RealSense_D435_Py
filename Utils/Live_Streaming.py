from Device.RealSense import RealSense
import numpy as np
import cv2
import os
import sys

if __name__ == "__main__":
    _root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(_root_path)
    os.chdir(_root_path)
    print('work_dir: ', _root_path)

    camera = RealSense('./Config/rs_d435.yaml')

    while True:
        frame = camera.get_frame()
        color_image = frame['color_image']
        depth_image_raw = frame['depth_image_raw']

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_raw, alpha=0.03), cv2.COLORMAP_JET)
        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            camera.stop()
            break
