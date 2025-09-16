import cv2
import numpy as np

# Try importing picamera (will fail on Windows)
try:
    import picamera
    import picamera.array
    USE_PICAMERA = True
except ImportError:
    USE_PICAMERA = False


def capture_image():
    """
    Capture a frame either from PiCamera (on Raspberry Pi) 
    or from the default webcam (on Windows/Linux).
    Returns: BGR image (numpy array)
    """
    if USE_PICAMERA:
        with picamera.PiCamera() as camera:
            with picamera.array.PiRGBArray(camera) as stream:
                camera.resolution = (640, 480)
                camera.brightness = 60
                camera.exposure_mode = 'off'
                camera.awb_mode = 'off'
                camera.awb_gains = (1.5, 1.5)

                camera.start_preview()
                camera.wait_recording(2)  # let camera adjust
                camera.capture(stream, format='bgr')
                camera.stop_preview()
                return stream.array
    else:
        cap = cv2.VideoCapture(0)  # webcam index
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        return frame


def process_image(image):
    """
    Process the captured image for Nile Red fluorescence detection.
    Returns: feature_data, mask, processed_image
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define fluorescence color ranges
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([45, 255, 255])

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # Build mask
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    combined_mask = mask_yellow + mask_red1 + mask_red2

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    feature_data = []
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # filter noise
            area = cv2.contourArea(contour)
            masked_area = cv2.bitwise_and(image, image, mask=combined_mask)
            mean_brightness = cv2.mean(masked_area, mask=combined_mask)[2]  # from V channel
            mean_hue = cv2.mean(hsv_image, mask=combined_mask)[0]
            feature_data.append([area, mean_brightness, mean_hue])

    return feature_data, combined_mask, image


def capture_and_process_image():
    img = capture_image()
    return process_image(img)


if __name__ == "__main__":
    features, mask, img = capture_and_process_image()
    print("Extracted features:", features)

    # Debug visualization
    cv2.imshow("Original", img)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
