import math
import warnings
from collections import deque
from io import BytesIO
from types import SimpleNamespace
from typing import NamedTuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from pyboy import PyBoy
from pyboy.utils import WindowEvent

from .memory_addresses import *

# Game Boy CPU clock divided by cycles per LCD frame (~59.73 frames per second)
FRAMES_PER_SECOND = 4194304 / 70224
PRESS_FRAMES = 8
MAX_ACTION_FRAMES = 600
# Stagnation patterns are re-evaluated every 12 controllable frames (~0.2 s), independent
# of action boundaries, so charges do not depend on action_freq.
STAGNATION_EVAL_FRAMES = 12

REWARD_COMPONENTS = (
    "unique_equipment",
    "equipment_upgrade",
    "house_exit",
    "new_room",
    "new_tile",
    "instrument",
    "story_milestone",
    "health_peak",
    "rupee_peak",
    "consumable_peak",
    "enemy_defeat",
    "critical_health",
    "stagnation",
    "death",
)
SMALL_REWARD_COMPONENTS = (
    "health_peak",
    "rupee_peak",
    "consumable_peak",
    "enemy_defeat",
)
RECURRING_PENALTY_COMPONENTS = ("critical_health", "stagnation")
PENALTY_COMPONENTS = RECURRING_PENALTY_COMPONENTS + ("death",)
STAGNATION_CATEGORIES = ("stationary", "pacing", "room_loop", "menu_loop")
LOCAL_STAGNATION_CATEGORIES = ("stationary", "pacing", "menu_loop")

# Signals that could not be verified against this ROM. They stay out of every total.
UNAVAILABLE_SIGNALS = {
    "enemy_defeat": "no verified RAM signal for entity identity, spawn generation, or defeat",
    "story_milestone": "no story flags beyond leaving Marin's house have been verified",
    "combat_activity": "no verified entity health/defeat signal to recognize useful combat",
    "puzzle_activity": "no verified puzzle-state signal to recognize useful puzzle progress",
}

REWARD_KEYS = (
    "unique_equipment",
    "equipment_upgrade",
    "house_exit",
    "new_room",
    "new_tile",
    "instrument",
    "story_milestone",
    "health_peak_per_heart",
    "rupee_peak_per_rupee",
    "consumable_peak_per_unit",
    "enemy_defeat",
    "critical_health_per_second",
    "death",
    "small_reward_budget",
    "recurring_penalty_budget",
    "tile_budget",
)
STAGNATION_KEYS = (
    "window_seconds",
    "base_rate",
    "room_loop_base_rate",
    "rate_step",
    "max_rate",
    "jitter_pixels",
    "bucket_pixels",
    "stationary_fraction",
    "min_repeats",
    "max_route_buckets",
    "new_bucket_tolerance",
    "max_room_loop_period",
)
EPISODE_KEYS = ("no_progress_seconds", "max_seconds")
OPTIONAL_SIGNAL_KEYS = ("enemy_defeat", "story_milestones")
GO_EXPLORE_KEYS = ("start_probability", "cell_pixels", "max_cells")


def _require_section(config, name, keys):
    section = config.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"config['{name}'] must be a dict with keys {list(keys)}")
    missing = [key for key in keys if key not in section]
    unknown = [key for key in section if key not in keys]
    if missing or unknown:
        raise ValueError(
            f"config['{name}'] has missing keys {missing} and unknown keys {unknown}"
        )
    return section


def _number(section_name, section, key, low=0.0, high=math.inf, integer=False):
    value = section[key]
    valid_type = isinstance(value, int) if integer else isinstance(value, (int, float))
    if (
        isinstance(value, bool)
        or not valid_type
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        kind = "an integer" if integer else "a finite number"
        raise ValueError(
            f"config['{section_name}']['{key}'] must be {kind} in [{low}, {high}], got {value!r}"
        )
    return value


def validate_config(config: dict) -> SimpleNamespace:
    """Validates reward, stagnation, and episode settings, failing on any invalid value."""
    action_freq = config.get("action_freq")
    if (
        isinstance(action_freq, bool)
        or not isinstance(action_freq, int)
        or not PRESS_FRAMES + 1 <= action_freq <= MAX_ACTION_FRAMES
    ):
        raise ValueError(
            f"action_freq must be an integer in [{PRESS_FRAMES + 1}, {MAX_ACTION_FRAMES}] "
            f"so the {PRESS_FRAMES}-frame press is released before the action ends, "
            f"got {action_freq!r}"
        )
    _number("config", config, "reward_scale", low=1e-9)

    rewards = _require_section(config, "rewards", REWARD_KEYS)
    for key in REWARD_KEYS:
        _number("rewards", rewards, key)

    stagnation = _require_section(config, "stagnation", STAGNATION_KEYS)
    _number("stagnation", stagnation, "window_seconds", low=1.0)
    for key in ("base_rate", "room_loop_base_rate", "rate_step"):
        _number("stagnation", stagnation, key)
    _number(
        "stagnation",
        stagnation,
        "max_rate",
        low=max(stagnation["base_rate"], stagnation["room_loop_base_rate"]),
    )
    _number("stagnation", stagnation, "jitter_pixels", high=255, integer=True)
    _number("stagnation", stagnation, "bucket_pixels", low=1, high=255, integer=True)
    _number("stagnation", stagnation, "stationary_fraction", low=0.5, high=1.0)
    _number("stagnation", stagnation, "min_repeats", low=2, high=100, integer=True)
    _number(
        "stagnation", stagnation, "max_route_buckets", low=2, high=1024, integer=True
    )
    _number("stagnation", stagnation, "new_bucket_tolerance", high=1024, integer=True)
    _number(
        "stagnation", stagnation, "max_room_loop_period", low=2, high=64, integer=True
    )

    episode = _require_section(config, "episode", EPISODE_KEYS)
    for key in EPISODE_KEYS:
        _number("episode", episode, key, low=1.0)

    optional = _require_section(config, "optional_signals", OPTIONAL_SIGNAL_KEYS)
    if optional["enemy_defeat"] is not False:
        raise ValueError(
            "optional_signals['enemy_defeat'] cannot be enabled: "
            + UNAVAILABLE_SIGNALS["enemy_defeat"]
        )
    if optional["story_milestones"] != []:
        raise ValueError(
            "optional_signals['story_milestones'] must be empty: "
            + UNAVAILABLE_SIGNALS["story_milestone"]
        )

    go_explore = _require_section(config, "go_explore", GO_EXPLORE_KEYS)
    _number("go_explore", go_explore, "start_probability", high=1.0)
    _number("go_explore", go_explore, "cell_pixels", low=8, high=256, integer=True)
    _number("go_explore", go_explore, "max_cells", low=1, high=100000, integer=True)

    window_frames = round(stagnation["window_seconds"] * FRAMES_PER_SECOND)
    return SimpleNamespace(
        action_freq=action_freq,
        reward_scale=config["reward_scale"],
        rewards=SimpleNamespace(**rewards),
        stagnation=SimpleNamespace(**stagnation, window_frames=window_frames),
        no_progress_frames=round(episode["no_progress_seconds"] * FRAMES_PER_SECOND),
        max_frames=round(episode["max_seconds"] * FRAMES_PER_SECOND),
        go_explore=SimpleNamespace(**go_explore),
    )


def decode_bcd(value: int, name: str) -> int:
    high, low = value >> 4, value & 0x0F
    if high > 9 or low > 9:
        raise ValueError(f"{name} holds invalid packed BCD 0x{value:02X}")
    return high * 10 + low


class FrameSample(NamedTuple):
    health: int
    gameplay_type: int
    gameplay_subtype: int
    motion: int
    room_transition: int
    dialog_state: int
    dialog_index: int
    window_y: int
    room: tuple
    x: int
    y: int

    @property
    def controllable(self) -> bool:
        # Dialogue and the open inventory stay controllable; only scripted motion,
        # map loading, room scrolling, and subscreen sliding are forced.
        return (
            self.gameplay_type == GAMEPLAY_TYPE_WORLD
            and self.gameplay_subtype == GAMEPLAY_SUBTYPE_PLAYING
            and self.motion == 0
            and self.room_transition == 0
            and self.window_y in (WINDOW_Y_HIDDEN, WINDOW_Y_INVENTORY_OPEN)
        )

    @property
    def menu_open(self) -> bool:
        return self.window_y == WINDOW_Y_INVENTORY_OPEN


class RewardLedger:
    """Episode-local cumulative credits and debits, in unscaled reward units."""

    def __init__(self, scale, small_budget, recurring_budget, tile_budget):
        self.scale = scale
        self.small_budget = small_budget
        self.recurring_budget = recurring_budget
        self.tile_budget = tile_budget
        self.credits = dict.fromkeys(REWARD_COMPONENTS, 0.0)
        self.debits = dict.fromkeys(REWARD_COMPONENTS, 0.0)
        self._step = dict.fromkeys(REWARD_COMPONENTS, 0.0)

    def small_remaining(self):
        used = sum(self.credits[name] for name in SMALL_REWARD_COMPONENTS)
        return max(0.0, self.small_budget - used)

    def tile_remaining(self):
        return max(0.0, self.tile_budget - self.credits["new_tile"])

    def recurring_remaining(self, component):
        return max(0.0, self.recurring_budget - self.debits[component])

    def credit(self, component, amount):
        if component in SMALL_REWARD_COMPONENTS:
            amount = min(amount, self.small_remaining())
        elif component == "new_tile":
            amount = min(amount, self.tile_remaining())
        if amount > 0:
            self.credits[component] += amount
            self._step[component] += amount

    def debit(self, component, amount):
        if component in RECURRING_PENALTY_COMPONENTS:
            amount = min(amount, self.recurring_remaining(component))
        if amount > 0:
            self.debits[component] += amount
            self._step[component] -= amount

    def fitness(self):
        return self.scale * (sum(self.credits.values()) - sum(self.debits.values()))

    def pop_step(self):
        deltas = {name: self.scale * value for name, value in self._step.items()}
        self._step = dict.fromkeys(REWARD_COMPONENTS, 0.0)
        return deltas

    def scaled(self, values):
        return {name: self.scale * value for name, value in values.items()}


class StagnationTracker:
    """Detects stagnation over a rolling window of controllable emulated frames."""

    def __init__(self, settings):
        self.s = settings
        self.window = settings.window_frames
        self.route = deque(maxlen=256)
        self.active = {}
        self.rate = 0.0
        self.pending_frames = 0
        self.charge = 0.0
        self.frames = dict.fromkeys(STAGNATION_CATEGORIES, 0)
        self.flush()

    def flush(self):
        """Clears room-local evidence after a scene change or verified useful activity."""
        self._settle()
        self.positions = deque(maxlen=self.window)
        self.seen_buckets = set()
        self.menu_sessions = deque()
        self.menu_opened = None
        for category in LOCAL_STAGNATION_CATEGORIES:
            self.active.pop(category, None)
        self.rate = self._rate()

    def clear(self):
        self.flush()
        self.active.clear()
        self.rate = 0.0

    def stop(self):
        """Ends recurring charges, e.g. on death, after settling earlier frames."""
        self._settle()
        self.active.clear()
        self.rate = 0.0

    def take_charge(self):
        """Returns the unscaled stagnation debit accumulated since the last call."""
        self._settle()
        charge, self.charge = self.charge, 0.0
        return charge

    def _settle(self):
        # Pending frames are charged at the rate set by the latest evaluation.
        if self.pending_frames:
            self.charge += self.rate * self.pending_frames / FRAMES_PER_SECOND
            for category in self.active:
                self.active[category] += self.pending_frames
                self.frames[category] += self.pending_frames
            self.pending_frames = 0

    def _rate(self):
        rates = (
            min(
                self.s.max_rate,
                self._base_rate(category) + self.s.rate_step * (f // self.window),
            )
            for category, f in self.active.items()
        )
        return max(rates, default=0.0)

    def _base_rate(self, category):
        if category == "room_loop":
            return self.s.room_loop_base_rate
        return self.s.base_rate

    def enter_room(self, t, room, was_new):
        self.route.append((t, room, was_new))
        self.flush()

    def add_frame(self, t, x, y, menu_open, read_equipment, last_progress_t):
        bucket = (x // self.s.bucket_pixels, y // self.s.bucket_pixels)
        is_new = bucket not in self.seen_buckets
        self.seen_buckets.add(bucket)
        self.positions.append((x, y, bucket, is_new))
        if menu_open and self.menu_opened is None:
            self.menu_opened = (t, read_equipment())
        elif not menu_open and self.menu_opened is not None:
            opened_at, before = self.menu_opened
            self.menu_sessions.append((opened_at, before, read_equipment()))
            self.menu_opened = None
        self.pending_frames += 1
        if t % STAGNATION_EVAL_FRAMES == 0:
            self._settle()
            found = self.candidates(t, last_progress_t)
            for category in list(self.active):
                if category not in found:
                    del self.active[category]
            for category in found:
                self.active.setdefault(category, 0)
            self.rate = self._rate()

    def _stationary(self, xs, ys):
        dx = np.abs(xs - np.median(xs))
        dy = np.abs(ys - np.median(ys))
        near = np.maximum(dx, dy) <= self.s.jitter_pixels
        return near.mean() >= self.s.stationary_fraction

    def _pacing(self):
        buckets = [position[2] for position in self.positions]
        new_buckets = sum(position[3] for position in self.positions)
        distinct = len(set(buckets))
        if new_buckets > self.s.new_bucket_tolerance:
            return False
        if not 2 <= distinct <= self.s.max_route_buckets:
            return False
        changes = sum(1 for a, b in zip(buckets, buckets[1:]) if a != b)
        return changes >= self.s.min_repeats * (distinct - 1)

    def _menu_loop(self, t):
        while self.menu_sessions and self.menu_sessions[0][0] < t - self.window:
            self.menu_sessions.popleft()
        sessions = len(self.menu_sessions)
        distinct = len({(before, after) for _, before, after in self.menu_sessions})
        return (
            sessions >= self.s.min_repeats and sessions >= self.s.min_repeats * distinct
        )

    def _room_loop(self, t, last_progress_t):
        route = [entry for entry in self.route if entry[0] >= last_progress_t]
        if not route or t - route[-1][0] > self.window:
            return False
        for period in range(2, self.s.max_room_loop_period + 1):
            # Extend the periodic run of previously visited rooms back as far as it holds.
            length = 0
            while length < len(route):
                entry = route[-1 - length]
                if entry[2]:
                    break
                if length >= period and entry[1] != route[-1 - length + period][1]:
                    break
                length += 1
            if length < period * self.s.min_repeats:
                continue
            if len({entry[1] for entry in route[-period:]}) < 2:
                continue
            if t - route[-length][0] >= self.window:
                return True
        return False

    def candidates(self, t, last_progress_t):
        found = set()
        if len(self.positions) >= self.window:
            xs = np.fromiter((p[0] for p in self.positions), dtype=np.int16)
            ys = np.fromiter((p[1] for p in self.positions), dtype=np.int16)
            if self._stationary(xs, ys):
                found.add("stationary")
            if self._pacing():
                found.add("pacing")
            if self._menu_loop(t):
                found.add("menu_loop")
        if self._room_loop(t, last_progress_t):
            found.add("room_loop")
        return found


class EpisodeState:
    """Everything that must be cleared and reseeded on reset."""

    def __init__(self, settings):
        self.ledger = RewardLedger(
            settings.reward_scale,
            settings.rewards.small_reward_budget,
            settings.rewards.recurring_penalty_budget,
            settings.rewards.tile_budget,
        )
        self.stagnation = StagnationTracker(settings.stagnation)
        self.frames = 0
        self.controllable_frames = 0
        self.progress_age_frames = 0
        self.last_progress_t = 0
        self.critical_frames = 0
        self.dead = False
        self.room = None
        self.visited_rooms = set()
        # Rooms known before reset or entered live; visited flags can be credited
        # during a forced transition before the live entry completes.
        self.entered_rooms = set()
        self.new_rooms = 0
        # Room-qualified position buckets reached this episode (exploration reward).
        self.visited_tiles = set()
        self.equipment = set()
        self.levels = {}
        self.instruments = set()
        self.peaks = {}
        self.dialog_open = False
        self.dialog_page = 0
        self.dialog_keys = set()
        self.house_exit_available = False
        self.house_exit_frame = None
        self.sword_frame = None
        self.instrument_count = 0
        self.progress = False
        self.health = 0
        self.last_sample = None
        self.start = "initial"
        self.dialog_waiting = False


class ZeldaGymEnv(gym.Env):

    def __init__(self, config: dict, debug=False, show_window=True):
        super().__init__()
        self.rom_path = config["rom_path"]

        assert self.rom_path is not None, "ROM path is required"

        self.settings = validate_config(config)

        self.pyboy = PyBoy(
            self.rom_path,
            window="SDL2" if show_window else "null",
            sound=config["game_with_sound"],
        )

        self._fitness = 0
        self._previous_fitness = 0
        self._initial_state: BytesIO | None = None
        self.debug = debug

        self.action_freq = self.settings.action_freq
        self.reward_scale = self.settings.reward_scale

        if not self.debug:
            self.pyboy.set_emulation_speed(0)

        down = (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN)
        left = (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT)
        right = (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT)
        up = (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP)
        a = (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A)
        b = (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B)
        start = (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START)
        # Each action presses its buttons together. Direction+B walks with the B item in
        # use, e.g. with the shield raised, which is needed to push sea urchins on the
        # beach (runtime-verified in room E1).
        actions = [
            (down,),
            (left,),
            (right,),
            (up,),
            (a,),
            (b,),
            (start,),
            (down, b),
            (left, b),
            (right, b),
            (up, b),
        ]
        self.valid_actions = [
            tuple(press for press, _ in buttons) for buttons in actions
        ]
        self.release_actions = [
            tuple(release for _, release in buttons) for buttons in actions
        ]

        self.observation_space = Dict(
            {
                "screen": Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8),
                # Non-screen values are scaled to [0, 1] so they do not swamp the
                # normalized image features.
                "current_room_layout": Box(
                    low=0, high=1, shape=(156,), dtype=np.float32
                ),
                "items_in_hand": Box(low=0, high=1, shape=(2,), dtype=np.float32),
                "items_in_inventory": Box(
                    low=0, high=1, shape=(len(ADDR_INVENTORY),), dtype=np.float32
                ),
                "health": Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "rupees": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            }
        )

        self.action_space = Discrete(len(self.valid_actions))

        self.availability = dict.fromkeys(REWARD_COMPONENTS, True)
        self.availability.update(dict.fromkeys(UNAVAILABLE_SIGNALS, False))
        self._warned_house_exit = False
        warnings.warn(
            "Disabled reward signals (excluded from all totals): "
            + "; ".join(f"{name}: {why}" for name, why in UNAVAILABLE_SIGNALS.items()),
            stacklevel=2,
        )

        self._ep = EpisodeState(self.settings)
        # Go-Explore archive of savestates; it persists across episodes by design.
        self.archive = {}

    def step(self, action):
        ep = self._ep
        self.run_action(action)

        self._process_durable_state()
        ep.ledger.debit("stagnation", ep.stagnation.take_charge())
        self._archive_current_cell()

        terminated = ep.dead
        truncation_reason = None
        if not terminated:
            if ep.progress_age_frames >= self.settings.no_progress_frames:
                truncation_reason = "no_progress"
            elif ep.frames >= self.settings.max_frames:
                truncation_reason = "max_duration"
        truncated = truncation_reason is not None

        self._previous_fitness = self._fitness
        self._fitness = ep.ledger.fitness()
        reward = self._fitness - self._previous_fitness

        observation = self._get_observation()
        info = self._build_info(
            ep.ledger.pop_step(), "death" if terminated else None, truncation_reason
        )

        return observation, reward, terminated, truncated, info

    def run_action(self, action):
        """Holds the button for PRESS_FRAMES, then releases it, sampling RAM every frame."""
        for event in self.valid_actions[action]:
            self.pyboy.send_input(event)
        for frame in range(self.action_freq):
            if frame == PRESS_FRAMES:
                for event in self.release_actions[action]:
                    self.pyboy.send_input(event)
            if not self.pyboy.tick(1, frame == self.action_freq - 1):
                raise RuntimeError("Emulation stopped during an action.")
            self._process_frame(self._read_frame())

    def _read_frame(self) -> FrameSample:
        memory = self.pyboy.memory
        indoor = memory[ADDR_IS_INDOOR]
        if indoor not in (0, 1):
            raise ValueError(f"Unsupported indoor flag 0x{indoor:02X}")
        room_number = memory[ADDR_MAP_ROOM]
        room = (1, memory[ADDR_MAP_ID], room_number) if indoor else (0, 0, room_number)
        health = memory[ADDR_CURRENT_HEALTH]
        if health > MAX_HEARTS * HEALTH_UNITS_PER_HEART:
            raise ValueError(f"Unsupported health value 0x{health:02X}")
        return FrameSample(
            health=health,
            gameplay_type=memory[ADDR_GAMEPLAY_TYPE],
            gameplay_subtype=memory[ADDR_GAMEPLAY_SUBTYPE],
            motion=memory[ADDR_LINK_MOTION_STATE],
            room_transition=memory[ADDR_ROOM_TRANSITION_STATE],
            dialog_state=memory[ADDR_DIALOG_STATE],
            dialog_index=memory[ADDR_DIALOG_INDEX],
            window_y=memory[ADDR_WINDOW_Y],
            room=room,
            x=memory[ADDR_LINK_X],
            y=memory[ADDR_LINK_Y],
        )

    def _process_frame(self, sample: FrameSample):
        ep = self._ep
        ep.frames += 1
        if ep.dead:
            return
        ep.last_sample = sample
        ep.health = sample.health
        if sample.health == 0:
            ep.dead = True
            ep.stagnation.stop()
            ep.ledger.debit("death", self.settings.rewards.death)
            return

        self._track_dialog(sample)
        if not sample.controllable:
            return

        ep.controllable_frames += 1
        ep.progress_age_frames += 1
        if sample.room != ep.room:
            self._enter_room(sample.room)
        tile = self._tile(sample)
        if tile not in ep.visited_tiles:
            ep.visited_tiles.add(tile)
            ep.ledger.credit("new_tile", self.settings.rewards.new_tile)

        if sample.health <= HEALTH_UNITS_PER_HEART:
            ep.critical_frames += 1
            ep.ledger.debit(
                "critical_health",
                self.settings.rewards.critical_health_per_second / FRAMES_PER_SECOND,
            )

        ep.stagnation.add_frame(
            ep.controllable_frames,
            sample.x,
            sample.y,
            sample.menu_open,
            self._read_equipment_slots,
            ep.last_progress_t,
        )

    def _track_dialog(self, sample: FrameSample):
        """Clears local stagnation evidence when a dialogue reaches a page not seen this episode."""
        ep = self._ep
        state = sample.dialog_state & DIALOG_STATE_MASK
        if state == 0:
            ep.dialog_open = False
            ep.dialog_waiting = False
            return
        if not ep.dialog_open:
            ep.dialog_open = True
            ep.dialog_page = 0
        waiting = state == DIALOG_STATE_WAITING
        if waiting and not ep.dialog_waiting:
            key = (sample.dialog_index, ep.dialog_page)
            ep.dialog_page += 1
            if key not in ep.dialog_keys:
                ep.dialog_keys.add(key)
                ep.stagnation.flush()
        ep.dialog_waiting = waiting

    def _tile(self, sample: FrameSample):
        size = self.settings.stagnation.bucket_pixels
        return (sample.room, sample.x // size, sample.y // size)

    def _enter_room(self, room):
        """Records a completed entry: the first controllable frame in a new room identity."""
        ep = self._ep
        rewards = self.settings.rewards
        ep.room = room
        was_new = room not in ep.entered_rooms
        ep.entered_rooms.add(room)
        if room not in ep.visited_rooms:
            ep.visited_rooms.add(room)
            ep.new_rooms += 1
            ep.ledger.credit("new_room", rewards.new_room)
            ep.progress = True
        if ep.house_exit_available and ep.house_exit_frame is None and room[0] == 0:
            ep.house_exit_frame = ep.frames
            ep.ledger.credit("house_exit", rewards.house_exit)
            ep.progress = True
        ep.stagnation.enter_room(ep.controllable_frames, room, was_new)
        if ep.progress:
            self._mark_progress()

    def _cell_key(self, sample: FrameSample):
        size = self.settings.go_explore.cell_pixels
        equipment = tuple(sorted(self._ep.equipment, key=str))
        return (
            sample.room,
            sample.x // size,
            sample.y // size,
            equipment,
            len(self._ep.instruments),
        )

    def _archive_current_cell(self):
        """Saves a Go-Explore start state the first time a controllable cell is reached."""
        settings = self.settings.go_explore
        ep = self._ep
        sample = ep.last_sample
        if settings.start_probability == 0 or ep.dead or sample is None:
            return
        if not sample.controllable or sample.dialog_state & DIALOG_STATE_MASK:
            return
        key = self._cell_key(sample)
        if key in self.archive:
            self.archive[key]["seen"] += 1
            return
        if len(self.archive) >= settings.max_cells:
            # Drop the cell that has already been explored from the most.
            del self.archive[max(self.archive, key=lambda k: self.archive[k]["chosen"])]
        state = BytesIO()
        self.pyboy.save_state(state)
        self.archive[key] = {"state": state.getvalue(), "chosen": 0, "seen": 1}

    def _choose_archive_cell(self):
        """Picks a cell, favoring those chosen least often (1/sqrt(1 + chosen))."""
        keys = list(self.archive)
        weights = np.array([1 / math.sqrt(1 + self.archive[k]["chosen"]) for k in keys])
        cell = self.archive[
            keys[self.np_random.choice(len(keys), p=weights / weights.sum())]
        ]
        cell["chosen"] += 1
        return cell

    def _mark_progress(self):
        ep = self._ep
        ep.progress = False
        ep.progress_age_frames = 0
        ep.last_progress_t = ep.controllable_frames
        ep.stagnation.clear()

    def _process_durable_state(self):
        ep = self._ep
        rewards = self.settings.rewards
        memory = self.pyboy.memory

        for item in self._read_equipment_ids() - ep.equipment:
            ep.equipment.add(item)
            ep.ledger.credit("unique_equipment", rewards.unique_equipment)
            ep.progress = True
            if item == ITEM_SWORD and ep.sword_frame is None:
                ep.sword_frame = ep.frames

        for address, level in self._read_levels().items():
            # Level 1 arrives with the base item, which already earns equipment credit.
            gained = level - max(ep.levels[address], 1)
            if gained > 0:
                ep.ledger.credit(
                    "equipment_upgrade", rewards.equipment_upgrade * gained
                )
                ep.progress = True
            ep.levels[address] = max(ep.levels[address], level)

        for instrument in self._read_instruments() - ep.instruments:
            ep.instruments.add(instrument)
            ep.instrument_count += 1
            ep.ledger.credit("instrument", rewards.instrument)
            ep.progress = True

        for room in self._read_visited_overworld() - ep.visited_rooms:
            ep.visited_rooms.add(room)
            ep.new_rooms += 1
            ep.ledger.credit("new_room", rewards.new_room)
            ep.progress = True

        for name, value in self._read_resources().items():
            if value > ep.peaks[name]:
                if name == "health":
                    component = "health_peak"
                    weight = rewards.health_peak_per_heart / HEALTH_UNITS_PER_HEART
                elif name == "rupees":
                    component, weight = "rupee_peak", rewards.rupee_peak_per_rupee
                else:
                    component, weight = (
                        "consumable_peak",
                        rewards.consumable_peak_per_unit,
                    )
                ep.ledger.credit(component, (value - ep.peaks[name]) * weight)
                ep.peaks[name] = value

        if ep.progress:
            self._mark_progress()

    def _read_equipment_slots(self):
        slots = tuple(
            self.pyboy.memory[address] for address in ADDR_HELD_ITEMS + ADDR_INVENTORY
        )
        for value in slots:
            if value > ITEM_ID_MAX:
                raise ValueError(f"Unsupported item ID 0x{value:02X}")
        return slots

    def _read_equipment_ids(self):
        equipment = {item for item in self._read_equipment_slots() if item}
        flippers = self.pyboy.memory[ADDR_FLIPPERS]
        if flippers not in (0, 1):
            raise ValueError(f"Unsupported flippers flag 0x{flippers:02X}")
        if flippers:
            equipment.add("flippers")
        return equipment

    def _read_levels(self):
        levels = {}
        for address in (ADDR_POWER_BRACELET_LEVEL, ADDR_SHIELD_LEVEL, ADDR_SWORD_LEVEL):
            level = self.pyboy.memory[address]
            if level > EQUIPMENT_LEVEL_MAX:
                raise ValueError(
                    f"Unsupported equipment level 0x{level:02X} at 0x{address:04X}"
                )
            levels[address] = level
        return levels

    def _read_instruments(self):
        owned = set()
        for index, address in enumerate(ADDR_DUNGEON_INSTRUMENTS):
            value = self.pyboy.memory[address]
            if value > 0x03:
                raise ValueError(
                    f"Unsupported instrument flag 0x{value:02X} at 0x{address:04X}"
                )
            if value & INSTRUMENT_OWNED_FLAG:
                owned.add(index)
        return owned

    def _read_visited_overworld(self):
        return {
            (0, 0, index)
            for index, address in enumerate(ADDR_WORLD_MAP_STATUS)
            if self.pyboy.memory[address] & WORLD_MAP_VISITED_FLAG
        }

    def _read_rupees(self):
        hundreds = decode_bcd(self.pyboy.memory[ADDR_RUPEES[0]], "Rupees (hundreds)")
        if hundreds > 9:
            raise ValueError(f"Unsupported rupee hundreds digit {hundreds}")
        return hundreds * 100 + decode_bcd(self.pyboy.memory[ADDR_RUPEES[1]], "Rupees")

    def _read_resources(self):
        memory = self.pyboy.memory
        return {
            "health": memory[ADDR_CURRENT_HEALTH],
            "rupees": self._read_rupees(),
            "arrows": decode_bcd(memory[ADDR_ARROWS], "Arrows"),
            "bombs": decode_bcd(memory[ADDR_BOMBS], "Bombs"),
            "powder": decode_bcd(memory[ADDR_MAGIC_POWDER], "Magic powder"),
        }

    def _build_info(self, step_deltas, termination_reason, truncation_reason):
        ep = self._ep
        ledger = ep.ledger

        def seconds(frames):
            return None if frames is None else frames / FRAMES_PER_SECOND

        return {
            "reward_components": step_deltas,
            "credits": ledger.scaled(ledger.credits),
            "debits": ledger.scaled(ledger.debits),
            "budgets_remaining": {
                "small_rewards": ledger.scale * ledger.small_remaining(),
                "new_tile": ledger.scale * ledger.tile_remaining(),
                **{
                    name: ledger.scale * ledger.recurring_remaining(name)
                    for name in RECURRING_PENALTY_COMPONENTS
                },
            },
            "critical_health": 1 <= ep.health <= HEALTH_UNITS_PER_HEART,
            "critical_health_seconds": seconds(ep.critical_frames),
            "stagnation_categories": sorted(ep.stagnation.active),
            "stagnation_rate": ledger.scale * ep.stagnation.rate,
            "stagnation_seconds": {
                name: seconds(frames) for name, frames in ep.stagnation.frames.items()
            },
            "progress_age_seconds": seconds(ep.progress_age_frames),
            "unique_rooms": len(ep.visited_rooms),
            "new_rooms": ep.new_rooms,
            "elapsed_frames": ep.frames,
            "controllable_frames": ep.controllable_frames,
            "termination_reason": termination_reason,
            "truncation_reason": truncation_reason,
            "milestones": {
                "house_exit_seconds": seconds(ep.house_exit_frame),
                "sword_seconds": seconds(ep.sword_frame),
                "instruments": ep.instrument_count,
            },
            "availability": dict(self.availability),
            "start": ep.start,
            "archive_cells": len(self.archive),
        }

    def start_sequence(self):
        save_slot = None
        name_entered = False

        for _ in range(300):
            gameplay_type = self.pyboy.memory[ADDR_GAMEPLAY_TYPE]
            gameplay_subtype = self.pyboy.memory[ADDR_GAMEPLAY_SUBTYPE]
            button = None

            if gameplay_type == 0x00:
                button = "start"
            elif (gameplay_type, gameplay_subtype) == (0x02, 0x07):
                if save_slot is None:
                    occupied_slots = self.pyboy.memory[ADDR_SAVE_FILES_EXIST]
                    save_slot = next(
                        (slot for slot in range(3) if not occupied_slots & (1 << slot)),
                        None,
                    )
                    if save_slot is None:
                        raise RuntimeError(
                            "Cannot start a new game: all three save slots are occupied."
                        )
                button = (
                    "start"
                    if self.pyboy.memory[ADDR_SAVE_SLOT] == save_slot
                    else "down"
                )
            elif (gameplay_type, gameplay_subtype) == (0x03, 0x02):
                if self.pyboy.memory[ADDR_NAME_ENTRY_POSITION] == 0:
                    button = "a"
                else:
                    button = "start"
                    name_entered = True
            elif (
                name_entered
                and (gameplay_type, gameplay_subtype) == (0x0B, 0x07)
                and self.pyboy.memory[ADDR_CURRENT_HEALTH] > 0
                and self.pyboy.memory[ADDR_DIALOG_STATE] != 0
            ):
                # Marin's first dialogue marks the end of Link's wake-up animation.
                initial_state = BytesIO()
                self.pyboy.save_state(initial_state)
                self._initial_state = initial_state
                return

            if button is not None:
                self.pyboy.button(button, 8)
            if not self.pyboy.tick(8, False) or not self.pyboy.tick(16, True):
                raise RuntimeError("Emulation stopped during the startup sequence.")

        raise RuntimeError(
            "Startup timed out before Link woke up "
            f"(gameplay type=0x{self.pyboy.memory[ADDR_GAMEPLAY_TYPE]:02X}, "
            f"subtype=0x{self.pyboy.memory[ADDR_GAMEPLAY_SUBTYPE]:02X})."
        )

    def reset(self, **kwargs):
        super().reset(**kwargs)

        start = "initial"
        if self._initial_state is None:
            self.start_sequence()
        elif (
            self.archive
            and self.np_random.random() < self.settings.go_explore.start_probability
        ):
            start = "archive"
            self.pyboy.load_state(BytesIO(self._choose_archive_cell()["state"]))
        else:
            self._initial_state.seek(0)
            self.pyboy.load_state(self._initial_state)

        self._fitness = 0
        self._previous_fitness = 0
        self._ep = ep = EpisodeState(self.settings)
        ep.start = start

        # Seed every baseline from the restored snapshot so initial conditions earn nothing.
        sample = self._read_frame()
        ep.health = sample.health
        ep.room = sample.room
        ep.visited_rooms = self._read_visited_overworld() | {sample.room}
        ep.entered_rooms = set(ep.visited_rooms)
        ep.visited_tiles = {self._tile(sample)}
        ep.equipment = self._read_equipment_ids()
        ep.levels = self._read_levels()
        ep.instruments = self._read_instruments()
        ep.peaks = self._read_resources()
        dialog_state = sample.dialog_state & DIALOG_STATE_MASK
        ep.dialog_open = dialog_state != 0
        ep.dialog_waiting = dialog_state == DIALOG_STATE_WAITING
        ep.house_exit_available = sample.room == MARIN_HOUSE_ROOM
        self.availability["house_exit"] = ep.house_exit_available
        if (
            start == "initial"
            and not ep.house_exit_available
            and not self._warned_house_exit
        ):
            self._warned_house_exit = True
            warnings.warn(
                "house_exit reward disabled: the reset snapshot is not in Marin's house "
                f"(room {sample.room})",
                stacklevel=2,
            )

        observation = self._get_observation()
        info = self._build_info(dict.fromkeys(REWARD_COMPONENTS, 0.0), None, None)
        return observation, info

    def render(self, mode="human"):
        pass

    def close(self):
        # Never write cartridge RAM: it would add save slots next to the ROM.
        self.pyboy.stop(save=False)

    def _get_observation(self):
        memory = self.pyboy.memory
        return {
            # PyBoy renders RGBA; the alpha channel is constant.
            "screen": np.array(self.pyboy.screen.ndarray[:, :, :3], dtype=np.uint8),
            "current_room_layout": np.array(
                [memory[address] / 255 for address in ADDR_CURRENTLY_LOADED_MAP],
                dtype=np.float32,
            ),
            "items_in_hand": np.array(
                [memory[address] / ITEM_ID_MAX for address in ADDR_HELD_ITEMS],
                dtype=np.float32,
            ),
            "items_in_inventory": np.array(
                [memory[address] / ITEM_ID_MAX for address in ADDR_INVENTORY],
                dtype=np.float32,
            ),
            "health": np.array(
                [memory[ADDR_CURRENT_HEALTH] / (MAX_HEARTS * HEALTH_UNITS_PER_HEART)],
                dtype=np.float32,
            ),
            "rupees": np.array([self._read_rupees() / 999], dtype=np.float32),
        }
