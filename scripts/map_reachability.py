"""Maps which rooms Link can reach without items, starting outside Marin's house.

Searches positions in each room by trying every direction from every reached position (via
emulator savestates), follows every exit that completes a room transition, and records the
room graph. Health is topped up after every move so enemies cannot end the search, and
dialogues are closed with A. Searched states are keyed by room and Link's position only, so
progress that changes other objects (such as pushing sea urchins with the shield, which takes
many presses) is not followed. Usage:

    uv run python scripts/map_reachability.py [--max-rooms 80]
"""

import argparse
import copy
import os
import sys
import time
import warnings
from collections import deque
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402
from config import memory_addresses as ram  # noqa: E402
from config.gym import ZeldaGymEnv  # noqa: E402

DIRECTIONS = {"N": "up", "S": "down", "W": "left", "E": "right"}
# Walking with B held raises the shield (it is equipped on B), which pushes sea urchins.
SHIELD_MOVES = {f"{d}+B": (button, "b") for d, button in DIRECTIONS.items()}


class Probe:
    def __init__(self):
        config = copy.deepcopy(main.config)
        config["game_with_sound"] = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.env = ZeldaGymEnv(config, show_window=False)
            self.env.reset()
        self.pyboy = self.env.pyboy
        self.memory = self.pyboy.memory

    def save(self):
        state = BytesIO()
        self.pyboy.save_state(state)
        return state.getvalue()

    def load(self, state):
        self.pyboy.load_state(BytesIO(state))

    def room(self):
        m = self.memory
        if m[ram.ADDR_IS_INDOOR]:
            return (1, m[ram.ADDR_MAP_ID], m[ram.ADDR_MAP_ROOM])
        return (0, 0, m[ram.ADDR_MAP_ROOM])

    def controllable(self):
        m = self.memory
        return (
            m[ram.ADDR_GAMEPLAY_TYPE] == ram.GAMEPLAY_TYPE_WORLD
            and m[ram.ADDR_GAMEPLAY_SUBTYPE] == ram.GAMEPLAY_SUBTYPE_PLAYING
            and m[ram.ADDR_LINK_MOTION_STATE] == 0
            and m[ram.ADDR_ROOM_TRANSITION_STATE] == 0
            and m[ram.ADDR_WINDOW_Y] == ram.WINDOW_Y_HIDDEN
        )

    def dialog_open(self):
        return self.memory[ram.ADDR_DIALOG_STATE] & ram.DIALOG_STATE_MASK != 0

    def heal(self):
        hearts = self.memory[ram.ADDR_MAX_HEALTH]
        self.memory[ram.ADDR_CURRENT_HEALTH] = hearts * ram.HEALTH_UNITS_PER_HEART

    def press(self, buttons, frames=8):
        buttons = (buttons,) if isinstance(buttons, str) else buttons
        for button in buttons:
            self.pyboy.button_press(button)
        self.pyboy.tick(frames, False)
        for button in buttons:
            self.pyboy.button_release(button)
        self.pyboy.tick(2, False)

    def settle(self, limit=900):
        """Waits out transitions and closes dialogues; False if control never returns."""
        quiet = 0
        for frame in range(limit):
            if self.dialog_open():
                quiet = 0
                if frame % 16 == 0:
                    self.press("a", 4)
                else:
                    self.pyboy.tick(1, False)
                continue
            if self.controllable():
                quiet += 1
                if quiet >= 4:
                    return True
            else:
                quiet = 0
            self.pyboy.tick(1, False)
        return False

    def goto(self, x, y, limit=200):
        for _ in range(limit):
            self.settle()
            dx = x - self.memory[ram.ADDR_LINK_X]
            dy = y - self.memory[ram.ADDR_LINK_Y]
            if abs(dx) <= 4 and abs(dy) <= 4:
                return
            if abs(dx) > 4:
                self.press("right" if dx > 0 else "left", min(8, abs(dx)))
            else:
                self.press("down" if dy > 0 else "up", min(8, abs(dy)))
        raise RuntimeError(f"could not reach ({x:02X}, {y:02X})")

    def leave_house_with_shield(self):
        """Replays the opening the trained agents learn: wake up, shield, leave."""
        self.settle(3000)
        self.goto(0x46, 0x70)
        for target in ((0x50, 0x70), (0x50, 0x54), (0x68, 0x54)):
            self.goto(*target)
        self.press("right")
        self.press("a")
        for _ in range(40):
            self.settle()
            self.pyboy.tick(30, False)
        if self.memory[ram.ADDR_SHIELD_LEVEL] != 1:
            raise RuntimeError("did not receive the shield")
        self.goto(0x50, 0x54)
        self.goto(0x50, 0x70)
        for _ in range(60):
            self.press("down")
            if self.room()[0] == 0 and self.settle():
                return
        raise RuntimeError("did not leave the house")

    def has_sword(self):
        slots = [self.memory[a] for a in ram.ADDR_HELD_ITEMS + ram.ADDR_INVENTORY]
        return ram.ITEM_SWORD in slots


def explore(
    probe, max_rooms, max_positions, screenshots=None, moves=DIRECTIONS, target=None
):
    start_room = probe.room()
    rooms = {start_room: probe.save()}
    edges = {}
    sword_room = None
    queue = deque([start_room])
    while queue and target not in rooms:
        room = queue.popleft()
        probe.load(rooms[room])
        seen = set()
        frontier = deque([probe.save()])
        while frontier and len(seen) < max_positions:
            state = frontier.popleft()
            for name, button in moves.items():
                probe.load(state)
                probe.press(button)
                probe.heal()
                if not probe.settle():
                    continue
                if probe.has_sword() and sword_room is None:
                    sword_room = probe.room()
                after_room = probe.room()
                if after_room != room:
                    direction = name.split("+")[0]
                    edges.setdefault(room, {}).setdefault(direction, set()).add(
                        after_room
                    )
                    if after_room not in rooms and len(rooms) < max_rooms:
                        rooms[after_room] = probe.save()
                        queue.append(after_room)
                        if screenshots:
                            probe.pyboy.tick(1, True)
                            path = os.path.join(
                                screenshots, f"{name_of(after_room)}.png"
                            )
                            probe.pyboy.screen.image.save(path.replace(":", "_"))
                    continue
                position = (
                    probe.memory[ram.ADDR_LINK_X],
                    probe.memory[ram.ADDR_LINK_Y],
                )
                key = (position[0] // 4, position[1] // 4)
                if key not in seen:
                    seen.add(key)
                    frontier.append(probe.save())
        print(
            f"  {name_of(room)}: {len(seen)} positions, "
            f"exits {format_exits(edges.get(room, {}))}",
            flush=True,
        )
    return start_room, rooms, edges, sword_room


def name_of(room):
    indoor, map_id, number = room
    return f"in:{map_id:02X}:{number:02X}" if indoor else f"ow:{number:02X}"


def format_exits(exits):
    return (
        ", ".join(
            f"{d}->{'/'.join(name_of(r) for r in sorted(rs))}"
            for d, rs in sorted(exits.items())
        )
        or "none"
    )


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-rooms", type=int, default=80)
    parser.add_argument("--max-positions", type=int, default=2000)
    parser.add_argument("--screenshots", help="save a PNG of each room on first entry")
    parser.add_argument(
        "--shield-push",
        action="store_true",
        help="also try moving with B held (shield raised), which the agent cannot do",
    )
    args = parser.parse_args()

    started = time.time()
    probe = Probe()
    probe.leave_house_with_shield()
    print(
        f"Start: {name_of(probe.room())} (shield, no sword; "
        f"shield push {'on' if args.shield_push else 'off'})",
        flush=True,
    )
    if args.screenshots:
        os.makedirs(args.screenshots, exist_ok=True)
    moves = {**DIRECTIONS, **SHIELD_MOVES} if args.shield_push else DIRECTIONS
    _, rooms, _, sword_room = explore(
        probe, args.max_rooms, args.max_positions, args.screenshots, moves
    )

    overworld = sorted(name_of(r) for r in rooms if r[0] == 0)
    indoor = sorted(name_of(r) for r in rooms if r[0] == 1)
    print(
        f"\nReachable without items ({len(rooms)} rooms, {time.time() - started:.0f} s):"
    )
    print(f"  overworld ({len(overworld)}): {' '.join(overworld)}")
    print(f"  indoor ({len(indoor)}): {' '.join(indoor)}")
    print(
        f"  sword picked up in: {name_of(sword_room) if sword_room else 'not reached'}"
    )
    if len(rooms) >= args.max_rooms:
        print("  (stopped at --max-rooms; more rooms may be reachable)")
    probe.pyboy.stop(save=False)


if __name__ == "__main__":
    main_cli()
