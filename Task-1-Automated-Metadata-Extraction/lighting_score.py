import cv2

def analyze_lighting(image_path):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = gray.mean()

    contrast = gray.std()

    if brightness < 80:
        lighting = "Dark"
    elif brightness < 170:
        lighting = "Normal"
    else:
        lighting = "Bright"

    return brightness, contrast, lighting