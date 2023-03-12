import cv2
from lib import calibration, video

# calibration.intrinsic()
# camera_matrix, dist_coeffs, R, tvecs = calibration.extrinsic()


live = video.openLive()

video.ballTracking(live)