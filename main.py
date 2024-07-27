from pyboy import PyBoy

from config.memory_addresses import ADDR_CURRENT_HEALTH, ADDR_POSITION_8X8


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
    try:
        pyboy = PyBoy("roms/ZeldaLinksAwakening.gb")
    except (FileNotFoundError):
        raise SystemExit("You should have your ROM in the roms/ folder")

    try:
        with open("roms/ZeldaLinksAwakening.gb.state", "rb") as state:
            pyboy.load_state(state)
    except (FileNotFoundError):
        pass

    while pyboy.tick():
        pyboy.tick()

        screen_image = pyboy.screen.image
        game_elements = get_game_elements(pyboy)

    pyboy.stop()
