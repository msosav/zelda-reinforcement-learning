# Addresses from https://datacrystal.romhacking.net/wiki/The_Legend_of_Zelda:_Link%27s_Awakening_(Game_Boy)/RAM_map

# Memory Addresses and Their Purpose

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

# Inventory and Items
ADDR_HELD_ITEMS = [0xDB00, 0xDB01]  # Your currently held items
# 01: Sword, 02: Bombs, 03: Power Bracelet
# 04: Shield, 05: Bow, 06: Hookshot, 07: Fire Rod
# 08: Pegasus Boots, 09: Ocarina, 0A: Feather
# 0B: Shovel, 0C: Magic Powder, 0D: Boomerang
ADDR_INVENTORY = [i for i in range(0xDB02, 0xDB0B)]
ADDR_FLIPPERS = 0xDB0C  # 01 = Have
ADDR_POTION = 0xDB0D  # 01 = Have
# Current item in trading game (01 = Yoshi, 0E = Magnifier)
ADDR_TRADING_GAME_ITEM = 0xDB0E
ADDR_SECRET_SHELLS = 0xDB0F  # Number of secret shells
ADDR_DUNGEON_ENTRANCE_KEYS = [i for i in range(0xDB10, 0xDB15)]  # 01 = Have
ADDR_GOLDEN_LEAVES = 0xDB15  # Number of golden leaves

# Dungeon Item Flags
ADDR_DUNGEON_ITEM_FLAGS = [
    [i for i in range(0xDB16 + 5 * d, 0xDB1B + 5 * d)] for d in range(10)
]  # 5 bytes per dungeon, 5th byte = quantity of keys

# Equipment
ADDR_POWER_BRACELET_LEVEL = 0xDB43
ADDR_SHIELD_LEVEL = 0xDB44
ADDR_SWORD_LEVEL = 0xDB4E

# Quantities
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
# Each increment of 08h = one full heart, 04h = one-half heart
ADDR_CURRENT_HEALTH = 0xDB5A
# Number of hearts in hex (max recommended: 0Eh = 14 hearts)
ADDR_MAX_HEALTH = 0xDB5B

# Rupees
ADDR_RUPEES = [0xDB5D, 0xDB5E]  # Number of rupees (e.g., 0999 for 999 rupees)

# Instruments
# 00 = No instrument, 03 = Have instrument
ADDR_DUNGEON_INSTRUMENTS = [i for i in range(0xDB65, 0xDB6D)]

# Dungeon Position
ADDR_DUNGEON_POSITION = 0xDBAE  # Position on the 8x8 dungeon grid
ADDR_DUNGEON_KEYS = 0xDBD0  # Quantity of keys in possession

# Save Slot Death Count
ADDR_DEATH_COUNT = [0xDB56 + i for i in range(3)]  # One byte per save slot
