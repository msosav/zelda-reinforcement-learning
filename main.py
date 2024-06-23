from pyboy import PyBoy
from pyboy.utils import WindowEvent

valid_buttons = ["a", "b", "start", "select", "left", "right", "up", "down"]


def load_state(path: str, pyboy: PyBoy) -> None:
    with open(path, "rb") as state:
        pyboy.load_state(state)


if __name__ == "__main__":
    pyboy = PyBoy("roms/ZeldaLinksAwakening.gb")

    load_state("roms/ZeldaLinksAwakening.gb.state", pyboy)

    while pyboy.tick():
        pyboy.button(valid_buttons[0])  # a
        pyboy.tick()
        pass

    pyboy.stop()
