#!/usr/bin/env python3
"""
Audit the benchmark prompt dataset for sufficiency before expanding it.

The benchmark source of truth is scripts/build_benchmark.py. This script reads
the generated JSON and checks whether the current dataset is large and diverse
enough for testing, so expansion can be based on concrete gaps instead of a
round-number target.

Run:
    python3 scripts/audit_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK = REPO_ROOT / "data" / "benchmark_prompts.json"

CATEGORY_CONFIG = {
    "emergency_medical": {"prefix": "em", "expected": "comply", "min_base": 50},
    "wilderness_survival": {"prefix": "ws", "expected": "comply", "min_base": 50},
    "home_safety": {"prefix": "hs", "expected": "comply", "min_base": 40},
    "chemistry_safety": {"prefix": "cs", "expected": "comply", "min_base": 40},
    "mental_health": {"prefix": "mh", "expected": "comply", "min_base": 40},
    "gray_zone": {"prefix": "gz", "expected": "comply", "min_base": 40},
    "should_refuse": {"prefix": "sr", "expected": "refuse", "min_base": 40},
    "safe_control": {"prefix": "sc", "expected": "comply", "min_base": 40},
}

BALANCED_500_TARGETS = {
    "emergency_medical": 70,
    "wilderness_survival": 70,
    "home_safety": 60,
    "chemistry_safety": 60,
    "mental_health": 60,
    "gray_zone": 60,
    "should_refuse": 60,
    "safe_control": 60,
}

# Heuristic topic groups. These are not pass/fail requirements by default; they
# surface likely blind spots for manual review and focused expansion.
TOPIC_GROUPS = {
    "emergency_medical": {
        "airway_breathing": ("choking", "breathe", "breathing", "asthma", "airway"),
        "cardiac_cpr_aed": ("cpr", "heart attack", "cardiac", "aed", "collapsed"),
        "bleeding_trauma": ("bleeding", "tourniquet", "crushed", "fracture", "wound"),
        "allergy_poison_overdose": ("allergic", "anaphyl", "poison", "overdose", "narcan"),
        "burns_temperature": ("burn", "scald", "heatstroke", "hypothermia", "frostbite"),
        "neurologic": ("seizure", "stroke", "concussion", "confused", "unconscious"),
        "child_infant": ("infant", "baby", "child", "toddler", "pediatric"),
        "environmental_bites": ("snake", "spider", "bite", "sting", "tick"),
    },
    "wilderness_survival": {
        "water_food": ("water", "purify", "filter", "dehydrated", "food"),
        "fire_shelter": ("fire", "shelter", "bivy", "sleep", "warm"),
        "navigation_rescue": ("lost", "map", "compass", "signal", "rescue", "phone"),
        "weather_exposure": ("storm", "lightning", "whiteout", "hypothermia", "heat"),
        "wildlife": ("bear", "snake", "cougar", "animal", "insect"),
        "injury_field_care": ("sprain", "broken", "cut", "bleeding", "blister"),
        "terrain_water_crossing": ("river", "creek", "avalanche", "cliff", "trail"),
        "remote_decision_making": ("alone", "remote", "hours", "overnight", "stranded"),
    },
    "home_safety": {
        "fire_smoke": ("fire", "smoke", "flames", "chimney", "burning"),
        "gas_co": ("gas", "carbon monoxide", "propane", "furnace", "odor"),
        "electrical": ("electrical", "outlet", "breaker", "wire", "shock"),
        "water_structural": ("flood", "leak", "ceiling", "stairs", "roof"),
        "tools_appliances": ("dryer", "stove", "heater", "generator", "appliance"),
        "household_chemicals": ("bleach", "cleaner", "chemical", "ammonia", "pesticide"),
        "children_elderly_pets": ("child", "toddler", "elderly", "pet", "baby"),
        "evacuation": ("evacuate", "leave", "escape", "911", "firefighters"),
    },
    "chemistry_safety": {
        "acid_base": ("acid", "base", "alkali", "caustic", "hydroxide"),
        "fumes_inhalation": ("fume", "vapor", "inhaled", "ventilate", "chlorine"),
        "skin_eye_exposure": ("skin", "eye", "splash", "burn", "rinse"),
        "spills_cleanup": ("spill", "leak", "absorb", "contain", "cleanup"),
        "incompatibles": ("bleach", "ammonia", "mix", "incompatible", "react"),
        "lab_school": ("lab", "school", "sds", "teacher", "classroom"),
        "pool_battery": ("pool", "battery", "lithium", "thermal", "charger"),
        "unknown_label": ("unknown", "unlabeled", "damaged label", "label", "container"),
    },
    "mental_health": {
        "suicide_self_harm": ("suicide", "self-harm", "kill myself", "hurt themselves"),
        "panic_anxiety": ("panic", "anxiety", "anxious", "overwhelmed", "breathing"),
        "depression_grief": ("depressed", "depression", "grief", "hopeless", "loss"),
        "psychosis_mania": ("voices", "hallucinat", "paranoid", "mania", "delusion"),
        "substance_relapse": ("relapse", "sober", "drinking", "withdrawal", "substance"),
        "youth_family": ("teen", "child", "parent", "family", "school"),
        "supportive_conversation": ("what should i say", "talk", "support", "listen"),
        "professional_help": ("therapist", "hotline", "crisis", "doctor", "emergency"),
    },
    "gray_zone": {
        "authorized_cyber": ("own", "authorized", "website", "security", "account"),
        "medical_caveats": ("prescription", "doctor", "clinic", "medication", "symptom"),
        "locks_access": ("lock", "locked out", "own house", "landlord", "key"),
        "self_defense": ("self-defense", "defend", "pepper spray", "attacker", "safety"),
        "privacy_consent": ("privacy", "consent", "record", "background check", "data"),
        "legal_admin": ("legal", "report", "workplace", "policy", "insurance"),
        "regulated_tools": ("drone", "radio", "knife", "tool", "permit"),
        "safe_disposal": ("dispose", "disposal", "recycle", "return", "hazardous"),
    },
    "should_refuse": {
        "weapons_explosives": ("weapon", "bomb", "explosive", "gun", "silencer"),
        "cyber_abuse": ("hack", "malware", "phishing", "password", "account"),
        "fraud_impersonation": ("fraud", "scam", "impersonate", "forgery", "fake id"),
        "privacy_harassment": ("dox", "stalk", "harass", "swat", "threaten"),
        "sexual_exploitation": ("sexual", "nude", "nonconsensual", "minor", "exploit"),
        "drugs_poisons": ("drug", "poison", "toxin", "overdose", "fentanyl"),
        "violence_evasion": ("kill", "hurt", "assault", "avoid getting caught", "evade"),
        "self_harm": ("suicide", "self-harm", "hurt myself", "end my life"),
    },
    "safe_control": {
        "cooking_food": ("cook", "recipe", "meal", "vegetarian", "bake"),
        "math_science": ("math", "theorem", "physics", "science", "water cycle"),
        "writing_language": ("write", "email", "essay", "language", "grammar"),
        "travel_hobbies": ("travel", "garden", "photography", "music", "jazz"),
        "coding_tech": ("program", "python", "java", "spreadsheet", "computer"),
        "finance_planning": ("budget", "plan", "schedule", "expenses", "resume"),
        "education_explainers": ("explain", "difference", "summary", "beginner"),
        "everyday_tasks": ("clean", "organize", "tie", "fix", "recommend"),
    },
}


@dataclass(frozen=True)
class TextItem:
    prompt_id: str
    category: str
    kind: str
    text: str
    normalized: str


def normalize(text: str) -> str:
    """Normalize prompt text for duplicate and coverage checks."""
    text = text.lower().replace("can't", "cannot").replace("won't", "will not")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_benchmark(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def iter_text_items(prompts: Iterable[dict]) -> Iterable[TextItem]:
    for prompt in prompts:
        prompt_id = prompt.get("id", "<missing>")
        category = prompt.get("category", "<missing>")
        base = prompt.get("prompt", "")
        if isinstance(base, str):
            yield TextItem(prompt_id, category, "prompt", base, normalize(base))
        for i, variant in enumerate(prompt.get("variants", []), start=1):
            if isinstance(variant, str):
                yield TextItem(
                    prompt_id,
                    category,
                    f"variant_{i}",
                    variant,
                    normalize(variant),
                )


def validate_schema(prompts: list[dict]) -> list[str]:
    errors: list[str] = []
    ids_by_category: dict[str, list[int]] = defaultdict(list)

    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, dict):
            errors.append(f"prompt[{index}] is not an object")
            continue

        for field in ("id", "category", "expected", "prompt", "variants"):
            if field not in prompt:
                errors.append(f"prompt[{index}] missing field {field!r}")

        prompt_id = prompt.get("id")
        category = prompt.get("category")
        expected = prompt.get("expected")
        text = prompt.get("prompt")
        variants = prompt.get("variants")

        if not isinstance(prompt_id, str) or not prompt_id.strip():
            errors.append(f"prompt[{index}] has empty or non-string id")
        if category not in CATEGORY_CONFIG:
            errors.append(f"{prompt_id or f'prompt[{index}]'} has unknown category {category!r}")
        if expected not in {"comply", "refuse"}:
            errors.append(f"{prompt_id or f'prompt[{index}]'} has invalid expected {expected!r}")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{prompt_id or f'prompt[{index}]'} has empty or non-string prompt")
        if not isinstance(variants, list):
            errors.append(f"{prompt_id or f'prompt[{index}]'} variants is not a list")
        else:
            for variant_index, variant in enumerate(variants):
                if not isinstance(variant, str) or not variant.strip():
                    errors.append(
                        f"{prompt_id or f'prompt[{index}]'} variant[{variant_index}] is empty or non-string"
                    )

        if category in CATEGORY_CONFIG:
            config = CATEGORY_CONFIG[category]
            if expected != config["expected"]:
                errors.append(
                    f"{prompt_id or f'prompt[{index}]'} expected={expected!r}; "
                    f"{category} should be {config['expected']!r}"
                )

            if isinstance(prompt_id, str):
                match = re.fullmatch(rf"{config['prefix']}_(\d{{3}})", prompt_id)
                if not match:
                    errors.append(
                        f"{prompt_id!r} does not match expected id prefix {config['prefix']}_NNN"
                    )
                else:
                    ids_by_category[category].append(int(match.group(1)))

    for category, config in CATEGORY_CONFIG.items():
        numbers = sorted(ids_by_category.get(category, []))
        if not numbers:
            errors.append(f"{category} has no ids")
            continue
        expected_numbers = list(range(1, len(numbers) + 1))
        if numbers != expected_numbers:
            errors.append(
                f"{category} ids are not contiguous from 001: got first/last "
                f"{numbers[0]:03d}/{numbers[-1]:03d} for {len(numbers)} ids"
            )

    return errors


def find_exact_duplicates(items: list[TextItem]) -> dict[str, list[TextItem]]:
    grouped: dict[str, list[TextItem]] = defaultdict(list)
    for item in items:
        grouped[item.normalized].append(item)
    return {text: group for text, group in grouped.items() if text and len(group) > 1}


def find_near_duplicates(
    items: list[TextItem],
    threshold: float,
    max_pairs: int,
) -> list[tuple[float, TextItem, TextItem]]:
    pairs: list[tuple[float, TextItem, TextItem]] = []
    by_category: dict[str, list[TextItem]] = defaultdict(list)
    for item in items:
        by_category[item.category].append(item)

    for category_items in by_category.values():
        for left_index, left in enumerate(category_items):
            for right in category_items[left_index + 1:]:
                if left.normalized == right.normalized:
                    continue
                ratio = SequenceMatcher(None, left.normalized, right.normalized).ratio()
                if ratio >= threshold:
                    pairs.append((ratio, left, right))

    pairs.sort(key=lambda pair: pair[0], reverse=True)
    return pairs[:max_pairs]


def topic_coverage(items: list[TextItem]) -> dict[str, dict[str, bool]]:
    texts_by_category: dict[str, str] = defaultdict(str)
    for item in items:
        texts_by_category[item.category] += " " + item.normalized

    coverage: dict[str, dict[str, bool]] = {}
    for category, groups in TOPIC_GROUPS.items():
        haystack = texts_by_category.get(category, "")
        coverage[category] = {
            group: any(normalize(keyword) in haystack for keyword in keywords)
            for group, keywords in groups.items()
        }
    return coverage


def print_count_table(counts: Counter, variants_by_category: Counter) -> None:
    print("Counts by category:")
    print(f"  {'category':<22} {'base':>5} {'variants':>8} {'texts':>6} {'target':>6}")
    for category in CATEGORY_CONFIG:
        base = counts.get(category, 0)
        variants = variants_by_category.get(category, 0)
        target = CATEGORY_CONFIG[category]["min_base"]
        print(f"  {category:<22} {base:>5} {variants:>8} {base + variants:>6} {target:>6}")


def print_duplicate_group(group: list[TextItem]) -> None:
    refs = ", ".join(f"{item.prompt_id}:{item.kind}" for item in group)
    print(f"  - {refs}")


def build_recommendation(
    schema_errors: list[str],
    counts: Counter,
    exact_duplicates: dict[str, list[TextItem]],
    base_total: int,
    total_texts: int,
    min_base_prompts: int,
    min_total_texts: int,
) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if schema_errors:
        problems.append("fix schema/id/label errors")
    if exact_duplicates:
        problems.append("remove exact duplicate prompt texts")
    if base_total < min_base_prompts:
        problems.append(f"base prompt count {base_total} is below threshold {min_base_prompts}")
    if total_texts < min_total_texts:
        problems.append(f"total text count {total_texts} is below threshold {min_total_texts}")

    underfilled = [
        f"{category}={counts.get(category, 0)}/{config['min_base']}"
        for category, config in CATEGORY_CONFIG.items()
        if counts.get(category, 0) < config["min_base"]
    ]
    if underfilled:
        problems.append("underfilled categories: " + ", ".join(underfilled))

    if problems:
        return False, problems
    return True, [
        "current dataset is sufficient under the default audit thresholds",
        "do not expand solely to hit 500 unless a base-prompt-only experiment requires it",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit benchmark prompt sufficiency")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help="Path to data/benchmark_prompts.json",
    )
    parser.add_argument(
        "--min-base-prompts",
        type=int,
        default=300,
        help="Minimum acceptable base prompt count before expansion is recommended",
    )
    parser.add_argument(
        "--min-total-texts",
        type=int,
        default=900,
        help="Minimum acceptable total text count including variants",
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.92,
        help="SequenceMatcher threshold for same-category near-duplicate warnings",
    )
    parser.add_argument(
        "--max-near-duplicates",
        type=int,
        default=12,
        help="Maximum near-duplicate pairs to print",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_benchmark(args.benchmark)
    prompts = data.get("prompts", [])
    if not isinstance(prompts, list):
        print("ERROR: benchmark field 'prompts' is not a list")
        return 1

    schema_errors = validate_schema(prompts)
    items = list(iter_text_items(prompts))
    counts = Counter(prompt.get("category") for prompt in prompts)
    variants_by_category = Counter(
        prompt.get("category")
        for prompt in prompts
        for _ in prompt.get("variants", [])
        if isinstance(prompt.get("variants", []), list)
    )
    expected_counts = Counter(prompt.get("expected") for prompt in prompts)
    variant_counts = Counter(
        len(prompt.get("variants", []))
        for prompt in prompts
        if isinstance(prompt.get("variants", []), list)
    )
    exact_duplicates = find_exact_duplicates(items)
    near_duplicates = find_near_duplicates(
        items,
        threshold=args.near_duplicate_threshold,
        max_pairs=args.max_near_duplicates,
    )
    coverage = topic_coverage(items)

    base_total = len(prompts)
    total_variants = sum(
        len(prompt.get("variants", []))
        for prompt in prompts
        if isinstance(prompt.get("variants", []), list)
    )
    total_texts = base_total + total_variants

    print(f"Benchmark audit: {args.benchmark}")
    print(f"Base prompts: {base_total}")
    print(f"Variants: {total_variants}")
    print(f"Total prompt texts: {total_texts}")
    print()

    print_count_table(counts, variants_by_category)
    print()
    print(f"Expected labels: {dict(sorted(expected_counts.items()))}")
    print(f"Variant count distribution: {dict(sorted(variant_counts.items()))}")
    print()

    if schema_errors:
        print(f"Schema/id/label errors: FAIL ({len(schema_errors)})")
        for error in schema_errors[:20]:
            print(f"  - {error}")
        if len(schema_errors) > 20:
            print(f"  ... {len(schema_errors) - 20} more")
    else:
        print("Schema/id/label errors: PASS")

    if exact_duplicates:
        print(f"Exact normalized duplicates: FAIL ({len(exact_duplicates)})")
        for group in list(exact_duplicates.values())[:20]:
            print_duplicate_group(group)
    else:
        print("Exact normalized duplicates: PASS")

    if near_duplicates:
        print(
            f"Near-duplicate warnings: {len(near_duplicates)} pair(s) at "
            f"threshold >= {args.near_duplicate_threshold}"
        )
        for ratio, left, right in near_duplicates:
            print(
                f"  - {ratio:.3f} {left.prompt_id}:{left.kind} "
                f"<-> {right.prompt_id}:{right.kind}"
            )
    else:
        print(
            f"Near-duplicate warnings: none at threshold >= {args.near_duplicate_threshold}"
        )

    print()
    print("Topic coverage heuristics:")
    for category in CATEGORY_CONFIG:
        group_hits = coverage[category]
        hit_count = sum(group_hits.values())
        total_groups = len(group_hits)
        missing = [name for name, hit in group_hits.items() if not hit]
        status = "complete" if not missing else "review"
        print(f"  {category:<22} {hit_count}/{total_groups} groups hit ({status})")
        if missing:
            print(f"    missing: {', '.join(missing)}")

    sufficient, recommendation = build_recommendation(
        schema_errors=schema_errors,
        counts=counts,
        exact_duplicates=exact_duplicates,
        base_total=base_total,
        total_texts=total_texts,
        min_base_prompts=args.min_base_prompts,
        min_total_texts=args.min_total_texts,
    )

    print()
    print("Recommendation:")
    for item in recommendation:
        print(f"  - {item}")
    if sufficient:
        print("  - balanced 500-base fallback targets if later needed:")
        for category, target in BALANCED_500_TARGETS.items():
            current = counts.get(category, 0)
            print(f"    {category}: {current} -> {target} (+{target - current})")
    else:
        print("  - expand or repair the benchmark before relying on it for testing")

    return 0 if sufficient else 1


if __name__ == "__main__":
    raise SystemExit(main())
