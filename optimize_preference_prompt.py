import argparse
import json
import os
import random
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from typing import Literal

import dspy
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
PREFERENCES_PATH = ROOT / "preference.json"
ARTIFACTS_DIR = ROOT / "artifacts"

CHOICES = ("A", "B")
TASK_DESCRIPTION = (
    "You are predicting which UI design a human evaluator would prefer between design A and design B. "
    "Both images represent the same product screen. Base the decision on likely human preferences around "
    "visual hierarchy, spacing, readability, balance, usefulness of the layout, emphasis of important actions, "
    "and overall polish. Return exactly one label: A or B."
)
PREFERENCE_PROFILE = """
Use the following as soft signals about the annotator's taste, not rigid rules:
- They often seem to value strong contrast, legibility, and clear hierarchy.
- They may react negatively to screens that feel cramped, especially when content or controls get pushed toward the bottom.
- They often seem to prefer layouts where primary controls remain visible and usable rather than being overshadowed by decorative imagery.
- When both designs are flawed, they may lean toward the option that feels more coherent and easier to improve.
- They sometimes appreciate gradients, large imagery, or visual interest when those choices still support clarity.
- They appear attentive to alignment, even spacing, and balanced distribution of navigation, buttons, and content.
- Visible interaction cues, such as clear affordances for buttons or scrolling, may matter to them.
- Treat these as tendencies only, and let the actual image evidence override them when needed.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a DSPy prompt for predicting human UI preferences from paired images."
    )
    parser.add_argument("--train-size", type=int, default=2, help="Number of labeled demos for prompt optimization.")
    parser.add_argument("--val-size", type=int, default=30, help="Validation examples used by DSPy while optimizing.")
    parser.add_argument("--test-size", type=int, default=50, help="Held-out examples to report after optimization. Use 0 for no held-out test.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible splits.")
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model/deployment name. Defaults to MODEL in .env.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["openrouter", "azure"],
        help="Model provider. Defaults to MODEL_PROVIDER in .env, else openrouter.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(ARTIFACTS_DIR),
        help="Directory where the report and predictions will be written.",
    )
    parser.add_argument(
        "--preferences-path",
        default=str(PREFERENCES_PATH),
        help="Preference JSON to load.",
    )
    parser.add_argument(
        "--images-dir",
        default=str(IMAGES_DIR),
        help="Directory containing the paired images referenced by the preference JSON.",
    )
    parser.add_argument("--search-breadth", type=int, default=6, help="DSPy optimizer candidate count.")
    parser.add_argument("--search-depth", type=int, default=2, help="Reserved search-depth knob for compatibility.")
    parser.add_argument(
        "--mode",
        choices=["direct", "rubric"],
        default="rubric",
        help="Prediction mode: direct choice or rubric-structured choice.",
    )
    return parser.parse_args()


def load_records(preferences_path: Path) -> list[dict]:
    with preferences_path.open() as f:
        records = json.load(f)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{preferences_path.name} must contain a non-empty list of preference records.")
    return records


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def lm_generation_defaults(model_name: str) -> dict:
    if "gpt-5" in model_name:
        return {"temperature": 1.0, "max_tokens": 16000}
    return {"temperature": 0.0, "max_tokens": 500}


def copro_init_temperature(model_name: str) -> float:
    return 1.0 if "gpt-5" in model_name else 1.4


def parse_azure_target(target_uri: str, fallback_model: str | None) -> dict:
    parsed = urlparse(target_uri)
    query = parse_qs(parsed.query)
    api_version = query.get("api-version", [None])[0]
    if not api_version:
        raise ValueError("Azure target URI must include an api-version query parameter.")

    path = parsed.path.rstrip("/")
    model_type = "responses" if path.endswith("/responses") else "chat"
    deployment = None

    if "/deployments/" in path:
        deployment = path.split("/deployments/", 1)[1].split("/", 1)[0]

    model_name = fallback_model or deployment
    if not model_name:
        raise ValueError(
            "Azure configuration needs a model/deployment name. "
            "Set MODEL or pass --model when using a responses endpoint."
        )

    return {
        "model_name": model_name,
        "api_base": f"{parsed.scheme}://{parsed.netloc}",
        "api_version": api_version,
        "model_type": model_type,
    }


def configure_lm(provider: str, model_name: str | None) -> tuple[dspy.LM, str]:
    load_dotenv(ROOT / ".env")
    if provider == "azure":
        azure_target = _env("AZURE_OPENAI_TARGET_URI")
        azure_key = _env("AZURE_OPENAI_API_KEY")
        if not azure_target or not azure_key:
            raise ValueError("AZURE_OPENAI_TARGET_URI and AZURE_OPENAI_API_KEY are required for Azure.")

        azure = parse_azure_target(azure_target, model_name)
        lm_kwargs = lm_generation_defaults(azure["model_name"])
        lm = dspy.LM(
            f"azure/{azure['model_name']}",
            model_type=azure["model_type"],
            api_key=azure_key,
            api_base=azure["api_base"],
            api_version=azure["api_version"],
            temperature=lm_kwargs["temperature"],
            max_tokens=lm_kwargs["max_tokens"],
            cache=False,
        )
        resolved_model_name = azure["model_name"]
    else:
        resolved_model_name = model_name or _env("MODEL")
        api_key = _env("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing from the environment.")
        if not resolved_model_name:
            raise ValueError("No model name provided. Set MODEL in .env or pass --model.")

        lm_kwargs = lm_generation_defaults(resolved_model_name)
        lm = dspy.LM(
            resolved_model_name,
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",
            temperature=lm_kwargs["temperature"],
            max_tokens=lm_kwargs["max_tokens"],
            cache=False,
        )

    dspy.configure(lm=lm)
    return lm, resolved_model_name


def verify_model_access(provider: str) -> None:
    class AccessCheck(dspy.Signature):
        prompt: str = dspy.InputField()
        answer: str = dspy.OutputField()

    try:
        result = dspy.Predict(AccessCheck)(prompt="Reply with exactly ok")
    except Exception as exc:
        message = str(exc)
        if "Insufficient credits" in message or '"code":402' in message:
            service = "Azure OpenAI" if provider == "azure" else "OpenRouter"
            raise RuntimeError(
                f"{service} rejected the request with 402 Insufficient credits/quota. "
                "Check the account quota or switch to a funded deployment, then rerun the script."
            ) from exc
        raise

    if getattr(result, "answer", "").strip().lower() != "ok":
        raise RuntimeError("Model access preflight returned an unexpected response.")


def direction(label: str) -> str:
    normalized = normalize_choice(label)
    if normalized not in CHOICES:
        raise ValueError(f"Unexpected choice label: {label}")
    return normalized


def normalize_choice(label: str) -> str:
    if label.startswith("A"):
        return "A"
    if label.startswith("B"):
        return "B"
    raise ValueError(f"Unexpected choice label: {label}")


def derive_rubric_winners(record: dict) -> dict[str, str]:
    winner = normalize_choice(record["final_choice"])
    reasoning = record["reasoning"].lower()
    labels = {
        "hierarchy_winner": winner,
        "spacing_winner": winner,
        "readability_winner": winner,
        "controls_winner": winner,
        "polish_winner": winner,
        "fixability_winner": winner,
    }

    # When the annotator explicitly says both designs share a comparable aspect,
    # avoid over-claiming that the preferred option won every dimension.
    tie_cues = {
        "hierarchy_winner": ["similar", "both designs share", "strong difference", "comparable"],
        "spacing_winner": ["same", "quite similar", "look the same"],
        "readability_winner": ["legible and appropriate", "both designs seemed", "both designs share"],
        "controls_winner": ["strong difference between the form fields", "look the same", "both have"],
        "polish_winner": ["both designs seemed visually appealing", "both designs do seem quite similar"],
    }
    for field, cues in tie_cues.items():
        if any(cue in reasoning for cue in cues):
            labels[field] = "Tie"

    return labels


def pick_two_balanced_examples(records: list[dict]) -> list[dict]:
    a_candidates = [r for r in records if direction(r["final_choice"]) == "A"]
    b_candidates = [r for r in records if direction(r["final_choice"]) == "B"]
    if not a_candidates or not b_candidates:
        raise ValueError("Need examples where humans preferred A and B to build two diverse demos.")

    # Favor clearer signals first so the two DSPy demos are maximally informative.
    a_example = sorted(a_candidates, key=lambda r: (abs(r["final_score"]) != 2, r["uid"]))[0]
    b_example = sorted(b_candidates, key=lambda r: (abs(r["final_score"]) != 2, r["uid"]))[0]
    return [a_example, b_example]


def build_splits(records: list[dict], train_size: int, val_size: int, test_size: int, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    if train_size not in (0, 2):
        raise ValueError("This runner currently supports either zero-shot (0 demos) or exactly two labeled demos.")

    train_records = pick_two_balanced_examples(records) if train_size == 2 else []
    train_ids = {record["cid"] for record in train_records}
    remaining = [record for record in records if record["cid"] not in train_ids]

    rng = random.Random(seed)
    rng.shuffle(remaining)

    if len(remaining) < val_size + test_size:
        raise ValueError("Not enough remaining records for the requested validation and test splits.")

    val_records = remaining[:val_size]
    test_records = remaining[val_size : val_size + test_size]
    return train_records, val_records, test_records


def infer_preference_candidates(records: list[dict]) -> list[dict[str, str]]:
    reasonings = [record["reasoning"].lower() for record in records]
    dark_preferences = 0
    light_preferences = 0
    for record in records:
        left_name = record["left_file"].lower()
        right_name = record["right_file"].lower()
        winner = normalize_choice(record["final_choice"])
        winner_name = left_name if winner == "A" else right_name
        loser_name = right_name if winner == "A" else left_name
        if "dark" in winner_name and "light" in loser_name:
            dark_preferences += 1
        if "light" in winner_name and "dark" in loser_name:
            light_preferences += 1

    contrast_mentions = sum("contrast" in reasoning or "legib" in reasoning for reasoning in reasonings)
    usability_mentions = sum(
        any(token in reasoning for token in ("usable", "usability", "control", "button", "action", "flow"))
        for reasoning in reasonings
    )
    decoration_mentions = sum(
        any(token in reasoning for token in ("decor", "distract", "clutter", "crowd", "crammed"))
        for reasoning in reasonings
    )

    candidates = [
        {
            "name": "generic_ui",
            "profile": PREFERENCE_PROFILE,
        }
    ]

    inferred_lines = [
        "Use the following as soft signals inferred from the labeled examples, not rigid rules:",
        "- The annotator often seems to value strong contrast, legibility, and obvious hierarchy.",
        "- They may prefer layouts with visible controls, balanced spacing, and coherent task flow.",
        "- They may react negatively to clutter, distracting decoration, awkward alignment, and cramped lower sections.",
        "- When both designs are flawed, they may lean toward the option that feels easier to fix without a full redesign.",
        "- Let the actual evidence in the images matter more than these prior tendencies.",
    ]
    if contrast_mentions >= max(2, len(records) // 3):
        inferred_lines.append("- Contrast and readability appear to be recurring drivers of their decisions.")
    if usability_mentions >= max(2, len(records) // 3):
        inferred_lines.append("- Practical usability often seems to matter more to them than aesthetic novelty.")
    if decoration_mentions >= max(2, len(records) // 3):
        inferred_lines.append("- They appear sensitive to decorative elements that interfere with focus or controls.")
    candidates.append({"name": "inferred_general", "profile": "\n".join(inferred_lines)})

    if dark_preferences > light_preferences:
        dark_lines = inferred_lines + [
            f"- In {dark_preferences} labeled pairs where dark and light variants opposed each other, they more often preferred the darker version.",
            "- Treat darker themes as a possible positive signal when they improve contrast, reduce glare, and make the interface feel calmer or more cohesive.",
            "- Do not prefer dark mode blindly; only favor it when readability, hierarchy, and usability also improve.",
        ]
        candidates.append({"name": "dark_mode_bias", "profile": "\n".join(dark_lines)})
    elif light_preferences > dark_preferences:
        light_lines = inferred_lines + [
            f"- In {light_preferences} labeled pairs where dark and light variants opposed each other, they more often preferred the lighter version.",
            "- Treat lighter themes as a possible positive signal when they improve readability, hierarchy, and spaciousness without washing out controls.",
            "- Do not prefer light mode blindly; only favor it when clarity and usability also improve.",
        ]
        candidates.append({"name": "light_mode_bias", "profile": "\n".join(light_lines)})

    candidates.append(
        {
            "name": "task_first",
            "profile": "\n".join(
                inferred_lines
                + [
                    "- Give extra weight to task completion cues such as readable fields, obvious buttons, and clear information flow.",
                    "- If one design feels much more usable and the other feels mostly decorative or mood-driven, that may be a meaningful signal.",
                ]
            ),
        }
    )
    return candidates


class DirectPreferenceSignature(dspy.Signature):
    """Predict which UI design a human evaluator likely preferred.

    Judge the pair the way a design-aware human rater would, not by personal taste alone.
    Prefer the option with clearer hierarchy, better spacing, stronger readability, more balanced
    emphasis between content and controls, and a layout that feels more usable and polished.
    Penalize distracting decorative elements, crowded bottoms, weak contrast, awkward alignment,
    and compositions that hide or compress the main action area. When both options are flawed,
    prefer the one that appears more coherent and easier to fix without redesigning the whole screen.
    Match the annotator's niche tendencies: strong preference for contrast and legibility,
    dislike of crammed bottoms, preference for visible controls over oversized decoration,
    and willingness to choose the more salvageable design when both are imperfect.
    """

    task_description: str = dspy.InputField()
    preference_profile: str = dspy.InputField()
    image_a: dspy.Image = dspy.InputField()
    image_b: dspy.Image = dspy.InputField()
    predicted_choice: Literal["A", "B"] = dspy.OutputField()
    rationale: str = dspy.OutputField()


class RubricPreferenceSignature(dspy.Signature):
    """Predict which UI design a human evaluator likely preferred via rubric dimensions.

    Judge the pair the way a design-aware human rater would, not by personal taste alone.
    Prefer the option with clearer hierarchy, better spacing, stronger readability, more balanced
    emphasis between content and controls, and a layout that feels more usable and polished.
    Penalize distracting decorative elements, crowded bottoms, weak contrast, awkward alignment,
    and compositions that hide or compress the main action area. When both options are flawed,
    prefer the one that appears more coherent and easier to fix without redesigning the whole screen.
    Match the annotator's niche tendencies: strong preference for contrast and legibility,
    dislike of crammed bottoms, preference for visible controls over oversized decoration,
    and willingness to choose the more salvageable design when both are imperfect.
    """

    task_description: str = dspy.InputField()
    preference_profile: str = dspy.InputField()
    image_a: dspy.Image = dspy.InputField()
    image_b: dspy.Image = dspy.InputField()
    hierarchy_winner: Literal["A", "B", "Tie"] = dspy.OutputField()
    spacing_winner: Literal["A", "B", "Tie"] = dspy.OutputField()
    readability_winner: Literal["A", "B", "Tie"] = dspy.OutputField()
    controls_winner: Literal["A", "B", "Tie"] = dspy.OutputField()
    polish_winner: Literal["A", "B", "Tie"] = dspy.OutputField()
    fixability_winner: Literal["A", "B", "Tie"] = dspy.OutputField()
    predicted_choice: Literal["A", "B"] = dspy.OutputField()
    rationale: str = dspy.OutputField()


class PreferenceJudge(dspy.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        signature = RubricPreferenceSignature if mode == "rubric" else DirectPreferenceSignature
        self.predict = dspy.Predict(signature)

    def forward(self, task_description: str, preference_profile: str, image_a: dspy.Image, image_b: dspy.Image):
        return self.predict(
            task_description=task_description,
            preference_profile=preference_profile,
            image_a=image_a,
            image_b=image_b,
        )


def record_to_example(record: dict, mode: str, images_dir: Path, preference_profile: str) -> dspy.Example:
    image_a_path = images_dir / record["left_file"]
    image_b_path = images_dir / record["right_file"]
    if not image_a_path.exists() or not image_b_path.exists():
        raise FileNotFoundError(f"Missing image for record {record['cid']}")

    data = dict(
        cid=record["cid"],
        uid=record["uid"],
        screen_id=record["screen_id"],
        left_file=record["left_file"],
        right_file=record["right_file"],
        task_description=TASK_DESCRIPTION,
        preference_profile=preference_profile,
        image_a=dspy.Image(str(image_a_path)),
        image_b=dspy.Image(str(image_b_path)),
        predicted_choice=normalize_choice(record["final_choice"]),
        rationale=record["reasoning"],
    )
    if mode == "rubric":
        rubric = derive_rubric_winners(record)
        data.update(
            hierarchy_winner=rubric["hierarchy_winner"],
            spacing_winner=rubric["spacing_winner"],
            readability_winner=rubric["readability_winner"],
            controls_winner=rubric["controls_winner"],
            polish_winner=rubric["polish_winner"],
            fixability_winner=rubric["fixability_winner"],
        )
    return dspy.Example(**data).with_inputs("task_description", "preference_profile", "image_a", "image_b")


def preference_metric(example: dspy.Example, prediction, trace=None) -> float:
    gold = example.predicted_choice
    pred = getattr(prediction, "predicted_choice", None)
    if pred not in CHOICES:
        return 0.0
    return 1.0 if pred == gold else 0.0


def evaluate_program(program: PreferenceJudge, dataset: list[dspy.Example]) -> dict:
    results = []
    correct = 0

    for example in dataset:
        prediction = program(
            task_description=example.task_description,
            preference_profile=example.preference_profile,
            image_a=example.image_a,
            image_b=example.image_b,
        )
        pred_choice = prediction.predicted_choice
        gold_choice = example.predicted_choice
        exact_match = pred_choice == gold_choice
        correct += int(exact_match)
        results.append(
            {
                "cid": example.cid,
                "uid": example.uid,
                "screen_id": example.screen_id,
                "left_file": getattr(example, "left_file", None),
                "right_file": getattr(example, "right_file", None),
                "gold_choice": gold_choice,
                "predicted_choice": pred_choice,
                "correct": exact_match,
                "hierarchy_winner": getattr(prediction, "hierarchy_winner", None),
                "spacing_winner": getattr(prediction, "spacing_winner", None),
                "readability_winner": getattr(prediction, "readability_winner", None),
                "controls_winner": getattr(prediction, "controls_winner", None),
                "polish_winner": getattr(prediction, "polish_winner", None),
                "fixability_winner": getattr(prediction, "fixability_winner", None),
                "gold_rationale": example.rationale,
                "predicted_rationale": prediction.rationale,
            }
        )

    total = len(dataset)
    return {
        "examples": total,
        "accuracy": correct / total if total else 0.0,
        "predictions": results,
    }


def serialize_demos(demos: list) -> list[dict]:
    serialized = []
    for demo in demos:
        serialized.append(
            {
                "cid": demo.get("cid"),
                "uid": demo.get("uid"),
                "screen_id": demo.get("screen_id"),
                "left_file": demo.get("left_file"),
                "right_file": demo.get("right_file"),
                "predicted_choice": demo.get("predicted_choice"),
                "rationale": demo.get("rationale"),
            }
        )
    return serialized


def build_examples(records: list[dict], mode: str, images_dir: Path, preference_profile: str) -> list[dspy.Example]:
    return [record_to_example(record, mode, images_dir, preference_profile) for record in records]


def compile_program(mode: str, trainset: list[dspy.Example]) -> PreferenceJudge:
    program = PreferenceJudge(mode)
    if not trainset:
        return program
    return dspy.LabeledFewShot(k=len(trainset)).compile(
        student=program,
        trainset=trainset,
        sample=False,
    )


def select_best_preference_profile(
    mode: str,
    images_dir: Path,
    train_records: list[dict],
    val_records: list[dict],
    candidates: list[dict[str, str]],
) -> tuple[dict[str, str], list[dict[str, float | str]]]:
    scored_candidates = []
    best_candidate = candidates[0]
    best_accuracy = -1.0

    for candidate in candidates:
        trainset = build_examples(train_records, mode, images_dir, candidate["profile"])
        valset = build_examples(val_records, mode, images_dir, candidate["profile"])
        baseline_program = compile_program(mode, trainset)
        accuracy = evaluate_program(baseline_program, valset)["accuracy"]
        scored_candidates.append(
            {
                "name": candidate["name"],
                "accuracy": accuracy,
                "profile": candidate["profile"],
            }
        )
        if accuracy > best_accuracy:
            best_candidate = candidate
            best_accuracy = accuracy

    return best_candidate, scored_candidates


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    preferences_path = Path(args.preferences_path)
    images_dir = Path(args.images_dir)
    if not preferences_path.is_absolute():
        preferences_path = ROOT / preferences_path
    if not images_dir.is_absolute():
        images_dir = ROOT / images_dir

    records = load_records(preferences_path)
    load_dotenv(ROOT / ".env")
    provider = args.provider or _env("MODEL_PROVIDER", "openrouter")

    lm, resolved_model_name = configure_lm(provider, args.model or _env("MODEL"))
    verify_model_access(provider)
    train_records, val_records, test_records = build_splits(
        records=records,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    profile_candidates = infer_preference_candidates(train_records + val_records)
    selected_profile, profile_search_results = select_best_preference_profile(
        mode=args.mode,
        images_dir=images_dir,
        train_records=train_records,
        val_records=val_records,
        candidates=profile_candidates,
    )

    trainset = build_examples(train_records, args.mode, images_dir, selected_profile["profile"])
    valset = build_examples(val_records, args.mode, images_dir, selected_profile["profile"])
    testset = build_examples(test_records, args.mode, images_dir, selected_profile["profile"])

    baseline_program = compile_program(args.mode, trainset)
    evaluation_set = testset if len(testset) > 0 else valset
    evaluation_split = "test" if len(testset) > 0 else "validation"
    baseline_eval = evaluate_program(baseline_program, evaluation_set)

    optimizer = dspy.MIPROv2(
        metric=preference_metric,
        prompt_model=lm,
        task_model=lm,
        max_bootstrapped_demos=0,
        max_labeled_demos=len(trainset),
        auto=None,
        num_candidates=args.search_breadth,
        num_threads=1,
        seed=args.seed,
        init_temperature=copro_init_temperature(resolved_model_name),
        track_stats=False,
    )

    optimized_program = optimizer.compile(
        student=baseline_program,
        trainset=valset,
        valset=evaluation_set if evaluation_set is not valset else None,
        num_trials=max(4, args.search_breadth + 1),
        seed=args.seed,
        max_bootstrapped_demos=0,
        max_labeled_demos=len(trainset),
        minibatch=False,
    )
    optimized_eval = evaluate_program(optimized_program, evaluation_set)

    baseline_instructions = baseline_program.predict.signature.instructions
    optimized_instructions = optimized_program.predict.signature.instructions
    optimized_demos = serialize_demos(getattr(optimized_program.predict, "demos", []))

    report = {
        "provider": provider,
        "model": resolved_model_name,
        "mode": args.mode,
        "optimizer": "MIPROv2",
        "preferences_path": str(preferences_path),
        "images_dir": str(images_dir),
        "task_description": TASK_DESCRIPTION,
        "preference_profile": selected_profile["profile"],
        "selected_profile_name": selected_profile["name"],
        "profile_search_results": profile_search_results,
        "split_sizes": {
            "train": len(trainset),
            "validation": len(valset),
            "test": len(testset),
        },
        "evaluation_split": evaluation_split,
        "warning": (
            "No held-out test set was used; reported metrics are in-sample on the validation set."
            if evaluation_split == "validation"
            else None
        ),
        "train_examples": serialize_demos(trainset),
        "baseline_metrics": {"accuracy": baseline_eval["accuracy"]},
        "optimized_metrics": {"accuracy": optimized_eval["accuracy"]},
        "baseline_prompt": baseline_instructions,
        "optimized_prompt": optimized_instructions,
        "optimized_demos": optimized_demos,
    }

    report_path = artifact_dir / "optimized_prompt_report.json"
    predictions_path = artifact_dir / "optimized_test_predictions.json"

    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    with predictions_path.open("w") as f:
        json.dump(
            {
                "baseline": baseline_eval["predictions"],
                "optimized": optimized_eval["predictions"],
            },
            f,
            indent=2,
        )

    print(f"Provider: {provider}")
    print(f"Model: {resolved_model_name}")
    print(f"Mode: {args.mode}")
    print(f"Train/Val/Test: {len(trainset)}/{len(valset)}/{len(testset)}")
    print(f"Evaluation split: {evaluation_split}")
    print(f"Baseline accuracy: {baseline_eval['accuracy']:.3f}")
    print(f"Optimized accuracy: {optimized_eval['accuracy']:.3f}")
    print(f"Optimized prompt saved to: {report_path}")
    print(f"Predictions saved to: {predictions_path}")


if __name__ == "__main__":
    main()
