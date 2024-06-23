from pyboy import PyBoy

if __name__ == "__main__":
    pyboy = PyBoy("roms/ZeldaLinksAwakening.gb")

    while pyboy.tick():
        pass

    pyboy.stop()
