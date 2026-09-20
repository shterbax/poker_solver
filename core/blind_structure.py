import re
from typing import Optional, Dict
from pydantic import BaseModel


# --- Данная функция должна быть на верхнем уровне файла ---
def parse_poker_number(val_str: str) -> int:
    """Конвертирует '100', '1.2K', '2.5M' в чистый int."""
    clean = val_str.upper().strip().replace(',', '')
    if 'M' in clean:
        return int(float(clean.replace('M', '')) * 1_000_000)
    if 'K' in clean:
        return int(float(clean.replace('K', '')) * 1_000)
    return int(clean)


class LevelData(BaseModel):
    level: int
    sb: int
    bb: int
    ante: int
    duration_min: int = 3


class BlindStructureManager:
    RAW_STRUCTURE = [
        [1, "50", "100", "13"], [2, "60", "120", "15"], [3, "75", "150", "19"],
        [4, "100", "200", "25"], [5, "120", "240", "30"], [6, "140", "280", "35"],
        [7, "160", "320", "40"], [8, "200", "400", "50"], [9, "250", "500", "63"],
        [10, "300", "600", "75"], [11, "350", "700", "88"], [12, "400", "800", "100"],
        [13, "500", "1K", "125"], [14, "600", "1.2K", "150"], [15, "700", "1.4K", "175"],
        [16, "800", "1.6K", "200"], [17, "1K", "2K", "250"], [18, "1.25K", "2.5K", "315"],
        [19, "1.5K", "3K", "375"], [20, "1.75K", "3.5K", "440"], [21, "2K", "4K", "500"],
        [22, "2.5K", "5K", "625"], [23, "3K", "6K", "750"], [24, "3.5K", "7K", "875"],
        [25, "4K", "8K", "1K"], [26, "5K", "10K", "1.25K"], [27, "6K", "12K", "1.5K"],
        [28, "7K", "14K", "1.75K"], [29, "8K", "16K", "2K"], [30, "10K", "20K", "2.5K"],
        [31, "12.5K", "25K", "3.12K"], [32, "15K", "30K", "3.75K"], [33, "17.5K", "35K", "4.37K"],
        [34, "20K", "40K", "5K"], [35, "25K", "50K", "6.25K"], [36, "30K", "60K", "7.5K"],
        [37, "35K", "70K", "8.75K"], [38, "40K", "80K", "10K"], [39, "50K", "100K", "12.5K"],
        [40, "60K", "120K", "15K"], [41, "70K", "140K", "17.5K"], [42, "80K", "160K", "20K"],
        [43, "100K", "200K", "25K"], [44, "125K", "250K", "31.25K"], [45, "150K", "300K", "37.5K"],
        [46, "175K", "350K", "43.75K"], [47, "200K", "400K", "50K"], [48, "250K", "500K", "62.5K"],
        [49, "300K", "600K", "75K"], [50, "350K", "700K", "87.5K"], [51, "400K", "800K", "100K"],
        [52, "500K", "1M", "125K"], [53, "625K", "1.25M", "156.25K"], [54, "750K", "1.5M", "187.5K"],
        [55, "875K", "1.75M", "218.75K"], [56, "1M", "2M", "250K"], [57, "1.25M", "2.5M", "312.5K"],
        [58, "1.5M", "3M", "375K"], [59, "1.75M", "3.5M", "437.5K"], [60, "2M", "4M", "500K"],
        [61, "2.5M", "5M", "625K"], [62, "3M", "6M", "750K"], [63, "3.5M", "7M", "875K"],
        [64, "4M", "8M", "1M"], [65, "5M", "10M", "1.25M"], [66, "6M", "12M", "1.5M"],
        [67, "7M", "14M", "1.75M"], [68, "8M", "16M", "2M"], [69, "9M", "18M", "2.25M"],
        [70, "10M", "20M", "2.5M"]
    ]

    def __init__(self):
        self.levels: Dict[int, LevelData] = {}
        self.bb_map: Dict[int, LevelData] = {}
        self._build_index()

    def _build_index(self):
        for lvl, sb_str, bb_str, ante_str in self.RAW_STRUCTURE:
            sb = parse_poker_number(sb_str)
            bb = parse_poker_number(bb_str)
            ante = parse_poker_number(ante_str)

            data = LevelData(level=lvl, sb=sb, bb=bb, ante=ante)
            self.levels[lvl] = data
            self.bb_map[bb] = data

    def match_level(self, parsed_bb: int) -> Optional[LevelData]:
        if parsed_bb in self.bb_map:
            return self.bb_map[parsed_bb]

        all_bbs = list(self.bb_map.keys())
        closest_bb = min(all_bbs, key=lambda x: abs(x - parsed_bb))
        if abs(closest_bb - parsed_bb) / closest_bb < 0.15:
            return self.bb_map[closest_bb]

        return None