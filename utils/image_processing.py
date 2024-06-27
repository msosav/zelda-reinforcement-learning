from typing import Union

import cv2
import numpy as np
from PIL import Image


def process_image(screen_image: Image.Image) -> Union[Image.Image, np.array]:
    """
    Process the given screen image.

    Args:
        screen_image (PIL.Image.Image): The input screen image.

    Returns:
        Union[PIL.Image.Image, np.array]: The processed image as a PIL Image and a NumPy array.
    """
    screen_np = np.array(screen_image)

    gray_image = cv2.cvtColor(screen_np, cv2.COLOR_BGR2GRAY)

    _, thresh_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    screen_np = np.array(screen_np)

    cv2.imwrite("captures/thresh.png", thresh_image)

    return (thresh_image, screen_np)


def detect_contours(
    thresh_image: Image.Image, screen_np: np.array
) -> Union[tuple, np.ndarray]:
    """
    Detects contours in a thresholded image and draws them on the original image.

    Args:
        thresh_image (PIL.Image.Image): The thresholded image.
        screen_np (numpy.ndarray): The original image as a numpy array.

    Returns:
        Union[tuple, numpy.ndarray]: A tuple containing the detected contours and the image with contours drawn on it.
    """
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = cv2.drawContours(screen_np.copy(), contours, -1, (0, 255, 0), 2)

    cv2.imwrite("captures/contour.png", contour_image)

    return (contours, contour_image)
