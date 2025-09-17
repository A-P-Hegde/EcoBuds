from time import sleep
from picamera import PiCamera

def capture_image(filename='photo.jpg'):
    # Initialize the PiCamera object
    camera = PiCamera()
    
    try:
        # Set resolution 
        camera.resolution = (1920, 1080)  # Full HD
        
        # Start the camera preview 
        camera.start_preview()
        print("Camera preview started. Preparing to capture...")
        
        # Wait for camera to adjust exposure
        sleep(2)
        
        # Capture the image and save it
        camera.capture(filename)
        print(f"Image captured and saved as {filename}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Stop the preview and release the camera resources
        camera.stop_preview()
        camera.close()
        print("Camera released.")

if __name__ == '__main__':
    capture_image('captured_image.jpg')
