import cv2
from lib import calibration, video

# calibration.intrinsic()
# camera_matrix, dist_coeffs, R, tvecs = calibration.extrinsic()


live = video.openLive(720, 60)
video.ballTracking(live)

# calibration.getImages(live)

# calibration.intrinsic()

# calibration.extrinsic(live)


# video.extract_green_region(live) 
# video.ballTracking(live)