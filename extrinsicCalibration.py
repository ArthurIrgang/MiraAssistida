from lib import video, calibration

live = video.openLive(720, 30)
calibration.extrinsic(live)