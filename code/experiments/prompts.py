from __future__ import annotations

from pathlib import Path
from typing import Any


DOMAIN_NOUNS = {
    "animals": "animal",
    "birds": "bird",
    "building": "building",
    "butterflies": "butterfly",
    "cakes": "cake",
    "cars": "car",
}

PROMPT_SETTINGS = {
    "snow": "in a snow-covered landscape during snowfall",
    "autumn": "in an autumn landscape with falling maple leaves",
    "moss": "in a moss-covered wetland",
    "desert": "in a sandy desert",
}

# Class nouns only: colour, material, style, or species would let text solve the routing task.
SOURCE_SUBJECTS = {
    "animals": {
        "cheetah": "an animal",
        "cow": "an animal",
        "giraffe": "an animal",
        "goat": "an animal",
        "horse": "an animal",
        "kangaroo": "an animal",
    },
    "birds": {f"bird{index}": "a bird" for index in range(5)},
    "building": {
        "basil": "a building",
        "brownstone": "a residential building",
        "colosseum": "a building",
        "house-modern": "a house",
        "house-square": "a small house",
    },
    "butterflies": {
        f"butterfly{index}": "a butterfly" for index in range(5)
    },
    "cakes": {
        "cake0": "a cake",
        "cake1": "a pastry",
        "cake2": "a cupcake",
        "cake3": "a slice of cake",
        "cake4": "a slice of cake",
    },
    "cars": {
        "car0": "a compact car",
        "car1": "a sedan",
        "car2": "a compact car",
        "car3": "a pickup truck",
        "car4": "a hatchback",
    },
}


def indefinite_article(noun: str) -> str:
    return "an" if noun[:1].lower() in "aeiou" else "a"


def source_subject(domain: str, source: str | Path) -> str:
    stem = Path(source).stem
    try:
        return SOURCE_SUBJECTS[domain][stem]
    except KeyError as error:
        raise ValueError(f"unsupported source: {domain}/{stem}") from error


def base_prompt(domain: str, prompt_key: str, source: str | Path) -> str:
    subject = source_subject(domain, source)
    try:
        setting = PROMPT_SETTINGS[prompt_key]
    except KeyError as error:
        raise ValueError(f"unsupported setting: {prompt_key}") from error
    return f"A realistic photograph of {subject} {setting}."


def generic_photo_prompt(domain: str) -> str:
    noun = DOMAIN_NOUNS[domain]
    return f"A photo of {indefinite_article(noun)} {noun}"


def source_photo_prompt(domain: str, source: str | Path) -> str:
    return f"A photo of {source_subject(domain, source)}"


def apply_base_prompts(cases: list[dict[str, Any]]) -> None:
    for case in cases:
        case["prompt"] = base_prompt(
            str(case["domain"]), str(case["prompt_key"]), str(case["source"])
        )


def readable_region(name: str) -> str:
    names = {
        "head_neck": "head and neck",
        "limbs_tail": "limbs and tail",
        "head_beak": "head and beak",
        "torso_tail_legs": "torso, tail, and legs",
        "forewing_left": "left forewing",
        "forewing_right": "right forewing",
        # Mask label colours must not leak into the text as appearance cues.
        "red_region": "first target region",
        "blue_region": "second target region",
        "green_region": "third target region",
    }
    return names.get(name, name.replace("_", " "))


def appearance_clause(
    target_parts: list[dict[str, str]],
    donor_parts: list[dict[str, str]],
    descriptions: dict[str, str],
) -> str:
    if not donor_parts:
        return ""
    clauses = []
    for target, donor in zip(target_parts, donor_parts):
        donor_path = str(donor["donor"])
        if donor_path not in descriptions:
            raise KeyError(f"missing appearance description for {donor_path}")
        clauses.append(
            f"the {readable_region(str(target['name']))} showing "
            f"{descriptions[donor_path]}"
        )
    if len(clauses) == 1:
        joined = clauses[0]
    elif len(clauses) == 2:
        joined = " and ".join(clauses)
    else:
        joined = ", ".join(clauses[:-1]) + ", and " + clauses[-1]
    return ", with " + joined


def appearance_prompt(base: str, clause: str) -> str:
    return base.rstrip().removesuffix(".") + clause + "."


def self_check() -> None:
    assert base_prompt("animals", "snow", "sources/animals/cheetah.png") == (
        "A realistic photograph of a cheetah in a snow-covered landscape "
        "during snowfall."
    )
    assert base_prompt("building", "autumn", "colosseum.png") == (
        "A realistic photograph of an amphitheater in an autumn landscape "
        "with falling maple leaves."
    )
    assert generic_photo_prompt("animals") == "A photo of an animal"
    assert source_photo_prompt("cakes", "cake1.png") == "A photo of a doughnut"
    assert "dessert" not in base_prompt("cakes", "desert", "cake1.png")
    assert appearance_prompt("A photograph.", ", with a blue detail") == (
        "A photograph, with a blue detail."
    )


if __name__ == "__main__":
    self_check()
    print("[prompts] self-check OK")
