from PIL import Image
from pyboy import PyBoy

from config.memory_addresses import ADDR_CURRENT_HEALTH, ADDR_POSITION_8X8
from utils.game_elements import detect_game_elements
from utils.image_processing import detect_contours, process_image


def load_state(path: str, pyboy: PyBoy) -> None:
    """
    Load the state of the game from a file.

    Args:
        path (str): The path to the file containing the game state.
        pyboy (PyBoy): The PyBoy instance representing the game.

    Returns:
        None
    """
    with open(path, "rb") as state:
        pyboy.load_state(state)


def get_game_elements(pyboy: PyBoy) -> dict:
    """
    Extracts game elements from a screen image.

    Args:
        screen_image (PIL.Image.Image): The screen image to process.

    Returns:
        dict: A dictionary containing the detected game elements.
    """
    game_elements = {}

    game_elements["current_health"] = pyboy.memory[ADDR_CURRENT_HEALTH] / 8
    game_elements["links_position"] = pyboy.memory[ADDR_POSITION_8X8]

    return game_elements


if __name__ == "__main__":
    pyboy = PyBoy("roms/ZeldaLinksAwakening.gb")

    load_state("roms/ZeldaLinksAwakening.gb.state", pyboy)

    while pyboy.tick():
        pyboy.tick()

        screen_image = pyboy.screen.image
        game_elements = get_game_elements(pyboy)

        print("Health:", game_elements["current_health"])
        print("Position:", game_elements["links_position"])

    pyboy.stop()
