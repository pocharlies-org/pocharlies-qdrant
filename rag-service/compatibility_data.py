"""Platform registry and upgrade type taxonomy for compatibility analysis."""

from typing import Dict, Any

# Extended from knowledge_synthesizer.py PLATFORMS (lines 27-63)
# Adding barrel_lengths, hop_type, and spring_type for physical compatibility checks
PLATFORMS: Dict[str, Dict[str, Any]] = {
    "srs-a2": {
        "name": "Silverback SRS A2/M2",
        "type": "sniper",
        "keywords": ["srs", "srs a2", "srs m2", "srs-a2", "srs-m2"],
        "brands": ["Silverback", "STALKER"],
        "barrel_lengths": [430, 510, 555],
        "hop_type": "vsr",
        "spring_type": "aeg",
    },
    "tac-41": {
        "name": "Silverback TAC-41",
        "type": "sniper",
        "keywords": ["tac-41", "tac41", "tac 41"],
        "brands": ["Silverback", "STALKER"],
        "barrel_lengths": [430, 510],
        "hop_type": "vsr",
        "spring_type": "aeg",
    },
    "vsr-10": {
        "name": "VSR-10 Platform",
        "type": "sniper",
        "keywords": ["vsr", "vsr-10", "vsr10"],
        "brands": ["Tokyo Marui", "Action Army", "Maple Leaf"],
        "barrel_lengths": [303, 430],
        "hop_type": "vsr",
        "spring_type": "vsr",
    },
    "glock": {
        "name": "Glock GBB",
        "type": "gbb-pistol",
        "keywords": ["glock", "g17", "g18c", "g19", "g-series", "g series"],
        "brands": ["Tokyo Marui", "Umarex", "WE", "VFC"],
        "barrel_lengths": [80, 85, 97, 113],
        "hop_type": "gbb",
        "spring_type": "gbb",
    },
    "mws": {
        "name": "Tokyo Marui MWS M4 GBBR",
        "type": "gbbr",
        "keywords": ["mws", "m4 gbb", "m4 gbbr", "tm mws"],
        "brands": ["Tokyo Marui", "Wii Tech"],
        "barrel_lengths": [260, 363, 410],
        "hop_type": "gbb",
        "spring_type": "gbb",
    },
    "aap-01": {
        "name": "Action Army AAP-01",
        "type": "gbb-pistol",
        "keywords": ["aap-01", "aap01", "aap 01", "aap-01c"],
        "brands": ["Action Army"],
        "barrel_lengths": [129, 200, 250],
        "hop_type": "gbb",
        "spring_type": "gbb",
    },
    "hi-capa": {
        "name": "Hi-Capa 5.1",
        "type": "gbb-pistol",
        # Require "hi-capa"/"hicapa" — "5.1" alone is too generic
        "keywords": ["hi-capa", "hicapa", "hi capa"],
        "brands": ["Tokyo Marui", "WE", "AW Custom"],
        "barrel_lengths": [112, 113],
        "hop_type": "gbb",
        "spring_type": "gbb",
    },
    "m4-aeg": {
        "name": "M4 AEG",
        "type": "aeg",
        "keywords": ["m4 aeg", "m4a1", "m16 aeg", "ar-15 aeg", "ar15 aeg"],
        "brands": ["Tokyo Marui", "G&G", "Specna Arms", "VFC", "Krytac"],
        "barrel_lengths": [229, 300, 363, 407, 455, 509],
        "hop_type": "aeg",
        "spring_type": "aeg",
    },
    "ak-aeg": {
        "name": "AK AEG",
        "type": "aeg",
        "keywords": ["ak-47", "ak47", "ak74", "ak-74", "ak aeg"],
        "brands": ["LCT", "E&L", "CYMA", "Tokyo Marui"],
        "barrel_lengths": [455, 500],
        "hop_type": "aeg",
        "spring_type": "aeg",
    },
    "mp5-aeg": {
        "name": "MP5 AEG",
        "type": "aeg",
        "keywords": ["mp5", "mp5a4", "mp5a5", "mp5sd"],
        "brands": ["Tokyo Marui", "CYMA", "VFC"],
        "barrel_lengths": [229],
        "hop_type": "aeg",
        "spring_type": "aeg",
    },
}

UPGRADE_TYPES: Dict[str, str] = {
    "inner-barrel": "Precision/tight bore inner barrels",
    "outer-barrel": "External barrels, extensions, threaded adapters",
    "hop-up": "Hop-up chambers, buckings, nubs, tensioners",
    "spring": "Main springs, valve springs, nozzle springs",
    "trigger": "Trigger units, speed triggers, CNC triggers",
    "cylinder": "Cylinder sets, cylinder heads, air nozzles",
    "piston": "Pistons, piston heads, piston teeth",
    "gearbox": "Gearbox shells, gear sets, ETU units",
    "motor": "Motors (high-torque, high-speed)",
    "body-kit": "Stocks, grips, handguards, body parts",
    "rail-mount": "Rails, mounts, adapters, scope rings",
    "suppressor": "Silencers, flash hiders, tracer units",
    "magazine": "Magazines, mag adapters, speed loaders",
    "gas-system": "Gas valves, NPAS, nozzles, routers",
    "optic": "Scopes, red dots, holographic sights",
    "bb": "BBs, ammunition",
    "battery": "Batteries, chargers",
    "tool": "Installation tools, maintenance items",
    "upgrade-kit": "Multi-part upgrade bundles",
}

# Categories where products are cross-platform (no specific compatibility)
CROSS_PLATFORM_CATEGORIES = {
    "bb", "battery", "charger", "tool", "maintenance",
    "clothing", "gear", "protection", "tactical-gear",
    "bag", "case", "target", "grenade", "smoke",
    "face-protection", "eye-protection", "gloves",
}

# Keywords that indicate a product IS a base platform (gun/replica)
BASE_PLATFORM_INDICATORS = [
    "full auto", "semi auto", "gbb pistol", "aeg rifle",
    "sniper rifle", "shotgun", "submachine", "bolt action",
    "electric blowback", "gas blowback", "spring powered",
    "replica", "airsoft gun",
]
