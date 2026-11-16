# Addresses from https://datacrystal.romhacking.net/wiki/The_Legend_of_Zelda:_Link%27s_Awakening_(Game_Boy)/RAM_map
# (revision oldid=70573). Entries marked "Runtime-verified" were checked with controlled PyBoy traces of
# roms/ZeldaLinksAwakening.gb (SHA-256 21f712e2...169f): leaving Marin's house, walking and pushing walls,
# crossing rooms, opening/closing the inventory, advancing dialogue, damage/heal/death, and subscreen rendering.

# Memory Addresses and Their Purpose

# Startup and menus
# 00: Intro/title, 02: File select, 03: Name entry, 0B: World
ADDR_GAMEPLAY_TYPE = 0xDB95
# File select 07: Ready, name entry 02: Ready, world 07: Playing
ADDR_GAMEPLAY_SUBTYPE = 0xDB96
ADDR_SAVE_SLOT = 0xDBA6  # 00-02: Save slots, 03: Copy/erase menu row
ADDR_SAVE_FILES_EXIST = 0xDBA7  # Bits 0-2 indicate which save slots have a name
ADDR_NAME_ENTRY_POSITION = 0xDBAA  # 00-04: Position of the next character in the name
ADDR_DIALOG_STATE = 0xC19F  # 00: No dialogue, nonzero: Dialogue box open
# Runtime-verified dialogue state (bit 7 only selects the bottom box, so mask with 0x7F):
# 01-05 opening, 06/07 printing, 09 waiting for a button, 0A/0B scrolling to the next page, 0C-0F closing
DIALOG_STATE_MASK = 0x7F
DIALOG_STATE_WAITING = 0x09
# Runtime-verified: identifies the open dialogue (01: Marin's wake-up text, 8D: Tarin, 91: shield message)
ADDR_DIALOG_INDEX = 0xC173

# Live Link/room state (runtime-verified)
ADDR_LINK_X = 0xFF98  # Link's X pixel position within the current room; changes about 1 px per frame
ADDR_LINK_Y = 0xFF99  # Link's Y pixel position within the current room; stays constant while pushing a wall
ADDR_MAP_ROOM = (
    0xFFF6  # Current room number (overworld rooms index ADDR_WORLD_MAP_STATUS)
)
ADDR_MAP_ID = 0xFFF7  # Current indoor map (10: Marin's house); 00 on the overworld
ADDR_IS_INDOOR = 0xDBA5  # 00: Overworld, 01: Indoor map (houses, caves, dungeons)
# Marin's house at the wake-up snapshot: (indoor, map, room)
MARIN_HOUSE_ROOM = (0x01, 0x10, 0xA3)
# 00: Player-controllable. Nonzero values are scripted motion (03: leaving through a door,
# 04: entering a map, 07: dying)
ADDR_LINK_MOTION_STATE = 0xC11C
# 00: No transition. 01-05 while the screen scrolls to a neighboring room (forced)
ADDR_ROOM_TRANSITION_STATE = 0xC124
# Window Y: 80 during normal play, 00 while the inventory subscreen is open,
# 08-78 while it slides open or closed (forced), FF on the game-over screen
ADDR_WINDOW_Y = 0xDB9A
WINDOW_Y_HIDDEN = 0x80
WINDOW_Y_INVENTORY_OPEN = 0x00
# Gameplay type/subtype for normal world gameplay (runtime-verified: subtype 00-06 while loading a map)
GAMEPLAY_TYPE_WORLD = 0x0B
GAMEPLAY_SUBTYPE_PLAYING = 0x07

# Destination data
ADDR_DESTINATION_BYTE_1 = 0xD401  # 00: Overworld, 01: Dungeon, 02: Side view area
# Values from 00 to 1F accepted. FF is Color Dungeon
ADDR_DESTINATION_BYTE_2 = 0xD402
# Room number. Must appear on map or it will lead to an empty room
ADDR_DESTINATION_BYTE_3 = 0xD403
ADDR_DESTINATION_COORD_X = 0xD404  # Destination X coordinate
ADDR_DESTINATION_COORD_Y = 0xD405  # Destination Y coordinate

# Map Data
ADDR_CURRENTLY_LOADED_MAP = [i for i in range(0xD700, 0xD79C)]
ADDR_WORLD_MAP_STATUS = [i for i in range(0xD800, 0xD900)]
# 00: Unexplored, 10: Changed from initial status (e.g., sword taken on the beach or dungeon opened with key)
# 20: Owl talked, 80: Visited
# Example: Visiting the first dungeon's screen (80) and opening it with the key (10) would put that byte at 90
# Flags combine, so test the visited bit. Runtime-verified: leaving Marin's house sets D8A2, walking south sets D8B2
WORLD_MAP_VISITED_FLAG = 0x80

# Inventory and Items
ADDR_HELD_ITEMS = [0xDB00, 0xDB01]  # Your currently held items
# 01: Sword, 02: Bombs, 03: Power Bracelet
# 04: Shield, 05: Bow, 06: Hookshot, 07: Fire Rod
# 08: Pegasus Boots, 09: Ocarina, 0A: Feather
# 0B: Shovel, 0C: Magic Powder, 0D: Boomerang
# 00: Empty slot. Runtime-verified: Tarin's shield arrives as 04 in DB00
ITEM_ID_MAX = 0x0D
ITEM_SWORD = 0x01
# Ten slots, DB02-DB0B inclusive (runtime-verified: DB0B renders as the last subscreen slot)
ADDR_INVENTORY = [i for i in range(0xDB02, 0xDB0C)]
ADDR_FLIPPERS = 0xDB0C  # 01 = Have
ADDR_POTION = 0xDB0D  # 01 = Have
# Current item in trading game (01 = Yoshi, 0E = Magnifier)
ADDR_TRADING_GAME_ITEM = 0xDB0E
ADDR_SECRET_SHELLS = 0xDB0F  # Number of secret shells
ADDR_DUNGEON_ENTRANCE_KEYS = [i for i in range(0xDB10, 0xDB15)]  # 01 = Have
ADDR_GOLDEN_LEAVES = 0xDB15  # Number of golden leaves

# Dungeon Item Flags: eight 5-byte records, DB16-DB3D inclusive; the 5th byte is the key count.
# Not used for rewards: the per-byte meaning of the first four bytes is not verified.
ADDR_DUNGEON_ITEM_FLAGS = [
    [i for i in range(0xDB16 + 5 * d, 0xDB1B + 5 * d)] for d in range(8)
]

# Equipment levels: 00 none, 01 base item, 02 upgraded (runtime-verified: Tarin's shield sets DB44 00 -> 01,
# and the subscreen renders L-2 for 02 in each byte)
ADDR_POWER_BRACELET_LEVEL = 0xDB43
ADDR_SHIELD_LEVEL = 0xDB44
ADDR_SWORD_LEVEL = 0xDB4E
EQUIPMENT_LEVEL_MAX = 2

# Quantities: packed BCD, one byte each (runtime-verified: 12 renders as "12" and 09 as "09"; maxima start at 20/30/30)
ADDR_ARROWS = 0xDB45  # Number of arrows
ADDR_BOMBS = 0xDB4D  # Number of bombs
ADDR_MAGIC_POWDER = 0xDB4C  # Magic powder quantity
ADDR_MAX_MAGIC_POWDER = 0xDB76
ADDR_MAX_BOMBS = 0xDB77
ADDR_MAX_ARROWS = 0xDB78

# Ocarina
ADDR_OCARINA_SONGS = 0xDB49  # 3-bit mask: 0 = No songs, 7 = All songs
ADDR_OCARINA_SELECTED_SONG = 0xDB4A

# Health
# Each increment of 08h = one full heart, 04h = one-half heart (runtime-verified: damage and healing change
# it one unit at a time; the wake-up snapshot holds 18h with three hearts)
ADDR_CURRENT_HEALTH = 0xDB5A
# Number of hearts in hex (max recommended: 0Eh = 14 hearts)
ADDR_MAX_HEALTH = 0xDB5B
HEALTH_UNITS_PER_HEART = 8
MAX_HEARTS = 14

# Rupees
# Packed BCD, hundreds digit first (runtime-verified: 0099 + 1 -> 0100, 0100 - 1 -> 0099, capped at 0999,
# with no invalid digits on any frame while the counter updates)
ADDR_RUPEES = [0xDB5D, 0xDB5E]

# Instruments
# 00 = No instrument, 03 = Have instrument. Runtime-verified: the subscreen renders 02 and 03 as owned
# and 01 as missing, so bit 1 marks ownership
ADDR_DUNGEON_INSTRUMENTS = [i for i in range(0xDB65, 0xDB6D)]
INSTRUMENT_OWNED_FLAG = 0x02

# Dungeon Position
ADDR_DUNGEON_POSITION = 0xDBAE  # Position on the 8x8 dungeon grid
ADDR_DUNGEON_KEYS = 0xDBD0  # Quantity of keys in possession

# Save Slot Death Count
ADDR_DEATH_COUNT = [0xDB56 + i for i in range(3)]  # One byte per save slot
