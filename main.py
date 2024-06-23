from pyboy import PyBoy
from pyboy.utils import WindowEvent


def load_state(path: str, pyboy: PyBoy) -> None:
    with open(path, "rb") as state:
        pyboy.load_state(state)


if __name__ == "__main__":
    pyboy = PyBoy("roms/ZeldaLinksAwakening.gb")

    load_state("roms/ZeldaLinksAwakening.gb.state", pyboy)

    while pyboy.tick():
        pass

    pyboy.stop()
