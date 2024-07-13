# Addresses from https://datacrystal.romhacking.net/wiki/The_Legend_of_Zelda:_Link%27s_Awakening_(Game_Boy)/RAM_map

ADDR_CURRENT_HEALTH = 0xDB5A
ADDR_MAX_HEALTH = 0xDB5B
ADDR_POSITION_8X8 = 0xDBAE

# 00: Unexplored, 10: Changed from initial status (for example sword taken on the beach or dungeon opened with key)
# 20: Owl talked, 80: Visited
# For example, visiting the first dungeon's screen (80) and opening it with the key (10) would put that byte at 90
ADDR_WORLD_MAP_STATUS = [i for i in range(0xD800, 0xD900)]
