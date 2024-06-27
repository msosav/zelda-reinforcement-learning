import cv2


def detect_game_elements(contours: tuple) -> dict:
    """
    Detects game elements from a list of contours.

    Args:
        contours (tuple): A tuple of contours.

    Returns:
        dict: A dictionary containing the detected game elements.
            The dictionary has the following structure:
            {
                "enemies": [(x, y, w, h), ...],
                "walls": [(x, y, w, h), ...],
                "link": (x, y, w, h)
            }
            - "enemies" is a list of enemy bounding boxes.
            - "walls" is a list of wall bounding boxes.
            - "link" is the bounding box of the player character.
    """

    def is_enemy(x, y, w, h):
        return w > 15 and h > 15

    def is_wall(x, y, w, h):
        return w > 18 and h > 18

    def is_link(x, y, w, h):
        return w > 12 and h > 12

    game_elements = {"enemies": [], "walls": [], "link": None}

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 10:
                if is_enemy(x, y, w, h):
                    game_elements["enemies"].append((x, y, w, h))
                elif is_wall(x, y, w, h):
                    game_elements["walls"].append((x, y, w, h))
                elif is_link(x, y, w, h):
                    game_elements["link"] = (x, y, w, h)

    return game_elements
