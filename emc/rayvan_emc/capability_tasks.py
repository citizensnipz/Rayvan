from __future__ import annotations

import json
import random
import string
from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping

import torch
from torch import Tensor

from .data import CharacterTokenizer, LanguageCorpus, TINY_OVERFIT_TEXTS
from .tokenization import TextTokenizer

CAPABILITY_GENERATOR_VERSION = "rayvan-capability-v2"
CAPABILITIES = (
    "language",
    "associative_recall",
    "fuzzy_recall",
    "selective_copying",
    "working_memory",
    "compression",
    "arithmetic",
    "symbolic",
    "program_execution",
    "stateful_action",
)
SURFACES = ("english", "structured", "json", "code", "symbolic")
DEFAULT_MIXTURE_WEIGHTS: dict[str, float] = {
    "language": 0.20,
    "associative_recall": 0.15,
    "fuzzy_recall": 0.10,
    "working_memory": 0.10,
    "selective_copying": 0.08,
    "compression": 0.07,
    "arithmetic": 0.10,
    "symbolic": 0.08,
    "program_execution": 0.07,
    "stateful_action": 0.05,
}
DEFAULT_HELD_OUT_COMBINATIONS = (
    ("associative_recall", "json"),
    ("working_memory", "symbolic"),
)
TOKEN_BUDGETS = {"smoke": 50_000, "quick": 250_000, "standard": 1_000_000}
DIAGNOSTIC_CHECKPOINTS = {
    "smoke": (),
    "quick": (),
    "standard": (100_000, 250_000, 500_000, 1_000_000),
}


@dataclass(frozen=True)
class DiagnosticMetadata:
    capability: str
    operation: str
    surface_format: str
    difficulty: int
    sequence_length: int
    distractor_count: int
    generator_seed: int
    split: str
    held_out_combination: bool = False
    generator_version: str = CAPABILITY_GENERATOR_VERSION


@dataclass(frozen=True)
class DiagnosticExample:
    prompt: str
    target: str
    diagnostic_metadata: DiagnosticMetadata
    intermediate_targets: tuple[str, ...] = ()

    @property
    def model_text(self) -> str:
        """The only serialized model input: no diagnostic fields are included."""
        return f"{self.prompt}{self.target}\n"

    def metadata_dict(self) -> dict[str, object]:
        return asdict(self.diagnostic_metadata)


@dataclass(frozen=True)
class CapabilitySuiteConfig:
    seed: int = 42
    mixture_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_MIXTURE_WEIGHTS)
    )
    held_out_combinations: tuple[tuple[str, str], ...] = (
        DEFAULT_HELD_OUT_COMBINATIONS
    )
    minimum_difficulty: int = 1
    maximum_difficulty: int = 4

    def __post_init__(self) -> None:
        if set(self.mixture_weights) != set(CAPABILITIES):
            raise ValueError("mixture_weights must contain every capability exactly once")
        if any(weight < 0 for weight in self.mixture_weights.values()):
            raise ValueError("mixture weights cannot be negative")
        if sum(self.mixture_weights.values()) <= 0:
            raise ValueError("mixture weights need positive mass")
        if not 1 <= self.minimum_difficulty <= self.maximum_difficulty:
            raise ValueError("invalid difficulty range")
        for capability, surface in self.held_out_combinations:
            if capability not in CAPABILITIES or surface not in SURFACES:
                raise ValueError(f"invalid held-out combination: {(capability, surface)!r}")


class CapabilityTaskSuite:
    """Deterministic task generation with labels kept outside model-visible text."""

    def __init__(self, config: CapabilitySuiteConfig | None = None) -> None:
        self.config = config or CapabilitySuiteConfig()

    def generate(
        self,
        capability: str,
        *,
        split: Literal["train", "validation", "evaluation"],
        index: int,
        surface: str | None = None,
        difficulty: int | None = None,
    ) -> DiagnosticExample:
        if capability not in CAPABILITIES:
            raise ValueError(f"unknown capability: {capability!r}")
        generator_seed = self._example_seed(split, index, CAPABILITIES.index(capability))
        rng = random.Random(generator_seed)
        resolved_difficulty = difficulty or (
            self.config.minimum_difficulty
            + index
            % (self.config.maximum_difficulty - self.config.minimum_difficulty + 1)
        )
        allowed = self._allowed_surfaces(capability, split)
        resolved_surface = surface or allowed[index % len(allowed)]
        if resolved_surface not in allowed:
            raise ValueError(
                f"{capability}/{resolved_surface} is excluded from split {split}"
            )
        generated = _GENERATORS[capability](rng, resolved_difficulty, resolved_surface)
        prompt, target, operation, distractors, intermediates = generated
        held_out = (capability, resolved_surface) in self.config.held_out_combinations
        metadata = DiagnosticMetadata(
            capability=capability,
            operation=operation,
            surface_format=resolved_surface,
            difficulty=resolved_difficulty,
            sequence_length=len(prompt) + len(target),
            distractor_count=distractors,
            generator_seed=generator_seed,
            split=split,
            held_out_combination=held_out,
        )
        return DiagnosticExample(prompt, target, metadata, intermediates)

    def mixed_examples(
        self,
        count: int,
        *,
        split: Literal["train", "validation", "evaluation"],
    ) -> tuple[DiagnosticExample, ...]:
        if count <= 0:
            raise ValueError("count must be positive")
        rng = random.Random(self.config.seed + _SPLIT_OFFSETS[split])
        weights = [self.config.mixture_weights[name] for name in CAPABILITIES]
        capabilities = rng.choices(CAPABILITIES, weights=weights, k=count)
        return tuple(
            self.generate(capability, split=split, index=index)
            for index, capability in enumerate(capabilities)
        )

    def balanced_evaluation(
        self,
        examples_per_capability: int,
        *,
        held_out_only: bool = False,
    ) -> tuple[DiagnosticExample, ...]:
        if examples_per_capability <= 0:
            raise ValueError("examples_per_capability must be positive")
        examples: list[DiagnosticExample] = []
        for capability_index, capability in enumerate(CAPABILITIES):
            held_surfaces = [
                surface
                for held_capability, surface in self.config.held_out_combinations
                if held_capability == capability
            ]
            for local_index in range(examples_per_capability):
                if held_out_only and held_surfaces:
                    surface = held_surfaces[local_index % len(held_surfaces)]
                elif held_out_only:
                    continue
                else:
                    surface = SURFACES[local_index % len(SURFACES)]
                    if surface not in _CAPABILITY_SURFACES[capability]:
                        allowed = _CAPABILITY_SURFACES[capability]
                        surface = allowed[local_index % len(allowed)]
                index = capability_index * 1_000_000 + local_index
                examples.append(
                    self.generate(
                        capability,
                        split="evaluation",
                        index=index,
                        surface=surface,
                    )
                )
        return tuple(examples)

    def _allowed_surfaces(self, capability: str, split: str) -> tuple[str, ...]:
        surfaces = _CAPABILITY_SURFACES[capability]
        if split != "train":
            return surfaces
        allowed = tuple(
            surface
            for surface in surfaces
            if (capability, surface) not in self.config.held_out_combinations
        )
        if not allowed:
            raise ValueError(f"all training surfaces held out for {capability}")
        return allowed

    def _example_seed(self, split: str, index: int, capability_index: int) -> int:
        return (
            self.config.seed
            + _SPLIT_OFFSETS[split]
            + index * 104_729
            + capability_index * 10_007
        )


class CapabilityCorpus:
    """Packed mixed-task stream satisfying the existing training batch contract."""

    def __init__(
        self,
        suite: CapabilityTaskSuite,
        *,
        train_examples: int = 8_000,
        validation_examples: int = 1_000,
        tokenizer: TextTokenizer | None = None,
    ) -> None:
        self.suite = suite
        self.train_examples = suite.mixed_examples(train_examples, split="train")
        self.validation_examples = suite.mixed_examples(
            validation_examples, split="validation"
        )
        self._corpus = LanguageCorpus.from_texts(
            (example.model_text for example in self.train_examples),
            (example.model_text for example in self.validation_examples),
            tokenizer=tokenizer or diagnostic_tokenizer(),
        )
        self.tokenizer = self._corpus.tokenizer

    def sample_batch(
        self,
        split: Literal["train", "validation"],
        batch_size: int,
        sequence_length: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        return self._corpus.sample_batch(
            split,
            batch_size,
            sequence_length,
            generator=generator,
            device=device,
        )

    def fixed_sequences(
        self, split: Literal["train", "validation"], sequence_length: int
    ) -> tuple[Tensor, Tensor]:
        return self._corpus.fixed_sequences(split, sequence_length)


def diagnostic_tokenizer() -> CharacterTokenizer:
    characters = string.ascii_letters + string.digits + string.punctuation
    characters += " \n\t→⇒"
    return CharacterTokenizer(characters)


def token_budget_for_preset(preset: str) -> int:
    try:
        return TOKEN_BUDGETS[preset]
    except KeyError as error:
        raise ValueError(f"unknown diagnostic budget: {preset!r}") from error


def _language(
    rng: random.Random, difficulty: int, surface: str
) -> tuple[str, str, str, int, tuple[str, ...]]:
    del surface
    story = " ".join(
        rng.choice(TINY_OVERFIT_TEXTS) for _ in range(difficulty + 1)
    )
    words = story.split()
    cut = len(words) // 2
    prompt = " ".join(words[:cut]) + " "
    target = " ".join(words[cut:])
    return prompt, target, "continuation", 0, ()


def _associative_recall(
    rng: random.Random, difficulty: int, surface: str
) -> tuple[str, str, str, int, tuple[str, ...]]:
    count = 2 ** (difficulty + 1)
    keys = _unique_codes(rng, "K", count)
    values = _variable_value_codes(rng, count, maximum_parts=difficulty)
    pairs = list(zip(keys, values, strict=True))
    rng.shuffle(pairs)
    query_key, target = rng.choice(pairs)
    if difficulty >= 2:
        target = _variable_value_codes(rng, 1, maximum_parts=difficulty)[0]
        original_index = next(
            index for index, (key, _) in enumerate(pairs) if key == query_key
        )
        insertion = rng.randrange(original_index + 1, len(pairs) + 1)
        pairs.insert(insertion, (query_key, target))
    prompt = _format_lookup(pairs, query_key, surface, inverse=False)
    return prompt, target, "exact_lookup", len(pairs) - 1, ()


def _fuzzy_recall(
    rng: random.Random, difficulty: int, surface: str
) -> tuple[str, str, str, int, tuple[str, ...]]:
    count = 2 + difficulty * 2
    owners = [
        f"{owner} {rng.choice(('unit', 'group', 'team'))}"
        for owner in _unique_codes(rng, "P", count)
    ]
    object_heads = _unique_codes(rng, "Q", count)
    object_tails = _unique_codes(rng, "R", count)
    objects = [
        f"{head}-{tail}" for head, tail in zip(object_heads, object_tails, strict=True)
    ]
    pairs = list(zip(owners, objects, strict=True))
    rng.shuffle(pairs)
    query_owner, query_object = rng.choice(pairs)
    prompt = _format_lookup(pairs, query_object, surface, inverse=True)
    return prompt, query_owner, "inverse_compositional_lookup", count - 1, ()


def _selective_copying(
    rng: random.Random, difficulty: int, surface: str
) -> tuple[str, str, str, int, tuple[str, ...]]:
    total = 4 + difficulty * 3
    selected = sorted(rng.sample(range(total), k=max(2, difficulty + 1)))
    values = [rng.choice(string.ascii_uppercase) + str(rng.randrange(10)) for _ in range(total)]
    if surface == "english":
        rows = [
            f"{'remember' if index in selected else 'skip'} {value}."
            for index, value in enumerate(values)
        ]
        prompt = "\n".join(rows) + "\nList remembered items: "
    elif surface == "json":
        rows = [{"use": index in selected, "value": value} for index, value in enumerate(values)]
        prompt = json.dumps(rows, separators=(",", ":")) + "\nselected="
    elif surface == "code":
        rows = [f"emit({value})" if index in selected else f"pass({value})" for index, value in enumerate(values)]
        prompt = ";".join(rows) + "\noutput="
    elif surface == "symbolic":
        rows = [f"{'+' if index in selected else '-'}{value}" for index, value in enumerate(values)]
        prompt = " ".join(rows) + "\n+sequence="
    else:
        rows = [f"{int(index in selected)}:{value}" for index, value in enumerate(values)]
        prompt = " ".join(rows) + "\nones="
    target = " ".join(values[index] for index in selected)
    return prompt, target, "selective_filter", total - len(selected), ()


def apply_state_updates(
    initial: int, operations: list[tuple[str, int]]
) -> tuple[int, tuple[int, ...]]:
    value = initial
    states: list[int] = []
    for operation, operand in operations:
        if operation == "add":
            value += operand
        elif operation == "subtract":
            value -= operand
        elif operation == "set":
            value = operand
        else:
            raise ValueError(f"unknown state operation: {operation!r}")
        states.append(value)
    return value, tuple(states)


def evaluate_arithmetic_expression(expression: str) -> int:
    position = 0

    def factor() -> int:
        nonlocal position
        if position >= len(expression):
            raise ValueError("unexpected end of arithmetic expression")
        if expression[position] == "-":
            position += 1
            return -factor()
        if expression[position] == "(":
            position += 1
            value = additive()
            if position >= len(expression) or expression[position] != ")":
                raise ValueError("expected closing parenthesis")
            position += 1
            return value
        start = position
        while position < len(expression) and expression[position].isdigit():
            position += 1
        if start == position:
            raise ValueError("expected integer")
        return int(expression[start:position])

    def multiplicative() -> int:
        nonlocal position
        value = factor()
        while position < len(expression) and expression[position] in "*/":
            operation = expression[position]
            position += 1
            operand = factor()
            if operation == "*":
                value *= operand
            elif operand == 0 or value % operand:
                raise ValueError("division must be integral and nonzero")
            else:
                value //= operand
        return value

    def additive() -> int:
        nonlocal position
        value = multiplicative()
        while position < len(expression) and expression[position] in "+-":
            operation = expression[position]
            position += 1
            operand = multiplicative()
            value = value + operand if operation == "+" else value - operand
        return value

    result = additive()
    if position != len(expression):
        raise ValueError("unexpected trailing arithmetic input")
    return result


def evaluate_arithmetic_comparison(expression: str) -> bool:
    for operator in ("==", ">=", "<=", ">", "<"):
        if operator not in expression:
            continue
        left, right = expression.split(operator, 1)
        lhs = evaluate_arithmetic_expression(left)
        rhs = evaluate_arithmetic_expression(right)
        return {
            "==": lhs == rhs,
            ">=": lhs >= rhs,
            "<=": lhs <= rhs,
            ">": lhs > rhs,
            "<": lhs < rhs,
        }[operator]
    raise ValueError("arithmetic comparison requires a comparison operator")


def execute_restricted_program(program: str) -> tuple[int, tuple[int, ...]]:
    lines = [line.strip() for line in program.splitlines() if line.strip()]
    if len(lines) != 5 or not lines[0].startswith("x="):
        raise ValueError("unsupported restricted program")
    value = int(lines[0].split("=", 1)[1])
    repeat = int(lines[1].removeprefix("repeat ").removesuffix(":"))
    update = lines[2].removeprefix("x=x")
    operation, operand = update[0], int(update[1:])
    states: list[int] = []
    for _ in range(repeat):
        value = value + operand if operation == "+" else value - operand
        states.append(value)
    condition, action = lines[3].removeprefix("if ").split(": x=x", 1)
    if ">" in condition:
        threshold = int(condition.split(">", 1)[1])
        applies = value > threshold
    elif "<=" in condition:
        threshold = int(condition.split("<=", 1)[1])
        applies = value <= threshold
    else:
        raise ValueError("unsupported branch condition")
    if applies:
        branch_operation, branch_operand = action[0], int(action[1:])
        value = value + branch_operand if branch_operation == "+" else value - branch_operand
    states.append(value)
    if lines[4] != "print x":
        raise ValueError("restricted program must end with print x")
    return value, tuple(states)


def _working_memory(
    rng: random.Random, difficulty: int, surface: str
) -> tuple[str, str, str, int, tuple[str, ...]]:
    initial = rng.randrange(0, 10)
    operations = [
        (rng.choice(("add", "subtract", "set")), rng.randrange(1, 7))
        for _ in range(2**difficulty)
    ]
    value, states = apply_state_updates(initial, operations)
    prompt = _format_state(initial, operations, surface)
    return (
        prompt,
        str(value),
        "mutable_state_tracking",
        0,
        tuple(str(state) for state in states),
    )


def _compression(
    rng: random.Random, difficulty: int, surface: str
) -> tuple[str, str, str, int, tuple[str, ...]]:
    people = _unique_codes(rng, "P", 3 + difficulty)
    objects = _unique_codes(rng, "O", 2 + difficulty)
    owners = {item: rng.choice(people) for item in objects}
    events: list[tuple[str, str]] = []
    for _ in range(3 + difficulty * 2):
        item = rng.choice(objects)
        owner = rng.choice(people)
        owners[item] = owner
        events.append((item, owner))
    query = rng.choice(objects)
    if surface == "english":
        body = "\n".join(f"{owner} receives {item}." for item, owner in events)
        prompt = f"{body}\nSummarize only the final owner of {query}: owner({query})="
    elif surface == "json":
        prompt = json.dumps(events, separators=(",", ":")) + f"\nfinal_owner[{query}]="
    elif surface == "code":
        prompt = "\n".join(f"owner[{item}]={owner}" for item, owner in events) + f"\nprint(owner[{query}])="
    elif surface == "symbolic":
        prompt = " ".join(f"{item}→{owner}" for item, owner in events) + f"\n{query}→"
    else:
        prompt = " ".join(f"{item}:{owner}" for item, owner in events) + f"\nlast({query})="
    return prompt, owners[query], "delayed_state_compression", len(events) - 1, ()


def _arithmetic(
    rng: random.Random, difficulty: int, surface: str
) -> tuple[str, str, str, int, tuple[str, ...]]:
    expression = str(rng.randrange(1, 10))
    operations: list[str] = []
    intermediates: list[str] = []
    for _ in range(2 ** (difficulty - 1)):
        operation = rng.choice(("+", "-", "*", "/"))
        current = evaluate_arithmetic_expression(expression)
        if operation == "/":
            divisors = [
                candidate
                for candidate in range(1, 6)
                if current == 0 or current % candidate == 0
            ]
            operand = rng.choice(divisors)
        else:
            operand = rng.randrange(1, 6)
        expression = f"({expression}{operation}{operand})"
        operations.append(operation)
        intermediates.append(str(evaluate_arithmetic_expression(expression)))
    if difficulty >= 2 and rng.random() < 0.25:
        first, second, third = (rng.randrange(1, 10) for _ in range(3))
        expression = f"{first}+{second}*{third}"
        operations = ["order_of_operations"]
    value = evaluate_arithmetic_expression(expression)
    if rng.random() < 0.25:
        comparator = rng.choice(("==", ">", "<", ">=", "<="))
        comparison = f"{expression}{comparator}{rng.randrange(-5, 30)}"
        target = "true" if evaluate_arithmetic_comparison(comparison) else "false"
        expression = comparison
        operation_name = "arithmetic_comparison"
    else:
        target = str(value)
        operation_name = "arithmetic_" + "".join(operations)
    if surface == "english":
        prompt = f"Compute {expression}. Answer: "
    elif surface == "json":
        prompt = json.dumps({"evaluate": expression}, separators=(",", ":")) + "\nresult="
    elif surface == "code":
        prompt = f"print({expression})\noutput="
    elif surface == "symbolic":
        prompt = f"{expression}="
    else:
        prompt = f"EXPR:{expression}\nVALUE:"
    return prompt, target, operation_name, 0, tuple(intermediates)


def _symbolic(
    rng: random.Random, difficulty: int, surface: str
) -> tuple[str, str, str, int, tuple[str, ...]]:
    if difficulty >= 4:
        symbols = rng.sample(list(string.ascii_uppercase), difficulty + 1)
        rules = list(zip(symbols, symbols[1:]))
        if surface == "english":
            body = " ".join(f"Replace {left} with {right}." for left, right in rules)
            prompt = f"{body} Start with {symbols[0]}. Final symbol: "
        elif surface == "json":
            prompt = json.dumps(
                {"rules": rules, "start": symbols[0]}, separators=(",", ":")
            ) + "\nresult="
        elif surface == "code":
            prompt = ";".join(f"rewrite({left},{right})" for left, right in rules)
            prompt += f"\napply_all({symbols[0]})="
        elif surface == "symbolic":
            prompt = " ".join(f"{left}→{right}" for left, right in rules)
            prompt += f"\n{symbols[0]}⇒"
        else:
            prompt = " ".join(f"{left}:{right}" for left, right in rules)
            prompt += f"\ntransform {symbols[0]}="
        return prompt, symbols[-1], "abstract_symbol_rewrite", 0, tuple(symbols[1:])
    if difficulty % 2:
        x = rng.randrange(1, 8)
        coefficient = rng.randrange(2, 6)
        offset = rng.randrange(1, 8)
        rhs = coefficient * x + offset
        prompt = f"{coefficient}x+{offset}={rhs}\nx="
        return prompt, str(x), "linear_equation", 0, ()
    value = rng.randrange(1, 8)
    rows = [f"a={value}"]
    intermediates: list[str] = []
    previous = "a"
    for index in range(difficulty):
        name = chr(ord("b") + index)
        delta = rng.randrange(1, 5)
        value += delta
        rows.append(f"{name}={previous}+{delta}")
        intermediates.append(str(value))
        previous = name
    separator = ";" if surface in {"structured", "symbolic"} else "\n"
    prompt = separator.join(rows) + f"{separator}{previous}="
    return prompt, str(value), "variable_binding", 0, tuple(intermediates)


def _program_execution(
    rng: random.Random, difficulty: int, surface: str
) -> tuple[str, str, str, int, tuple[str, ...]]:
    del surface
    initial = rng.randrange(0, 8)
    repeat = 2 ** (difficulty - 1)
    delta = rng.randrange(1, 5)
    threshold = rng.randrange(5, 18)
    repeated_value = initial + repeat * delta
    branch = rng.randrange(1, 5)
    if repeated_value > threshold:
        branch_text = f"if x>{threshold}: x=x-{branch}"
    else:
        branch_text = f"if x<={threshold}: x=x+{branch}"
    program = (
        f"x={initial}\n"
        f"repeat {repeat}:\n x=x+{delta}\n"
        f"{branch_text}\nprint x"
    )
    value, states = execute_restricted_program(program)
    prompt = program + "\noutput="
    return (
        prompt,
        str(value),
        "restricted_program_execution",
        0,
        tuple(str(state) for state in states),
    )


def _stateful_action(
    rng: random.Random, difficulty: int, surface: str
) -> tuple[str, str, str, int, tuple[str, ...]]:
    environment = ("inventory", "key_value", "queue", "navigation")[difficulty % 4]
    if environment == "inventory":
        item = "key" + str(rng.randrange(10))
        room = rng.choice(("kitchen", "hall", "study"))
        prompt = f"inventory=[]; room={room}\npick_up({item}); move(hall); drop({item})\nwhere({item})="
        target = "hall"
        states = (f"inventory=[{item}]", "room=hall", "inventory=[]")
    elif environment == "key_value":
        key = "k" + str(rng.randrange(10))
        first, second = rng.randrange(10), rng.randrange(10)
        prompt = f"store={{}}\nput({key},{first}); put({key},{second}); get({key})="
        target = str(second)
        states = (f"{key}={first}", f"{key}={second}")
    elif environment == "queue":
        items = _unique_codes(rng, "T", 2 + difficulty)
        prompt = "queue=[]\n" + "; ".join(f"push({item})" for item in items)
        prompt += "; pop()\nfront="
        target = items[1]
        states = tuple(items)
    else:
        locations = ("A", "B", "C", "D")
        steps = 1 + difficulty
        index = 0
        actions = []
        for _ in range(steps):
            index = (index + 1) % len(locations)
            actions.append(f"move({locations[index]})")
        prompt = f"location={locations[0]}\n" + "; ".join(actions) + "\nlocation="
        target = locations[index]
        states = tuple(locations[1 : steps + 1])
    if surface == "json":
        prompt = json.dumps({"environment": environment, "trace": prompt}, separators=(",", ":")) + "\nanswer="
    elif surface == "english":
        prompt = "Execute these stateful actions.\n" + prompt
    return prompt, target, f"action_{environment}", 0, states


def _format_lookup(
    pairs: list[tuple[str, str]], query: str, surface: str, *, inverse: bool
) -> str:
    if surface == "english":
        if inverse:
            rows = [f"{left} owns object {right}." for left, right in pairs]
            return "\n".join(rows) + f"\nWho owns object {query}? "
        rows = [f"{left} maps to {right}." for left, right in pairs]
        return "\n".join(rows) + f"\nValue for {query}? "
    if surface == "json":
        mapping = dict(pairs)
        operation = "inverse_lookup" if inverse else "lookup"
        return json.dumps(mapping, separators=(",", ":")) + f"\n{operation}({query})="
    if surface == "code":
        name = "owner" if inverse else "table"
        rows = [f'{name}["{left}"]="{right}"' for left, right in pairs]
        operation = "inverse_lookup" if inverse else "lookup"
        return "\n".join(rows) + f'\n{operation}("{query}")='
    if surface == "symbolic":
        rows = [f"{left}→{right}" for left, right in pairs]
        operation = "inv" if inverse else "get"
        return " ".join(rows) + f"\n{operation}({query})="
    rows = [f"{left}:{right}" for left, right in pairs]
    operation = "inverse" if inverse else "query"
    return "\n".join(rows) + f"\n{operation} {query}="


def _format_state(initial: int, operations: list[tuple[str, int]], surface: str) -> str:
    if surface == "english":
        rows = [f"The score starts at {initial}."]
        rows.extend(f"{operation} {operand}." for operation, operand in operations)
        return "\n".join(rows) + "\nWhat is the final score? "
    if surface == "json":
        return json.dumps(
            {"initial": initial, "updates": operations}, separators=(",", ":")
        ) + "\nfinal="
    if surface == "code":
        rows = [f"score={initial}"]
        for operation, operand in operations:
            symbol = {"add": "+=", "subtract": "-=", "set": "="}[operation]
            rows.append(f"score{symbol}{operand}")
        return "\n".join(rows) + "\nprint(score)="
    if surface == "symbolic":
        symbols = {"add": "+", "subtract": "-", "set": "="}
        rows = [str(initial), *(f"{symbols[op]}{value}" for op, value in operations)]
        return " ".join(rows) + " → ?="
    return f"state:{initial} " + " ".join(f"{op}:{value}" for op, value in operations) + "\nstate="




def _variable_value_codes(
    rng: random.Random, count: int, *, maximum_parts: int
) -> list[str]:
    values: set[str] = set()
    alphabet = string.ascii_uppercase + string.digits
    while len(values) < count:
        parts = 1 + rng.randrange(maximum_parts)
        values.add(
            "V" + "-".join(
                "".join(rng.choice(alphabet) for _ in range(1 + rng.randrange(3)))
                for _ in range(parts)
            )
        )
    return sorted(values)

def _unique_codes(
    rng: random.Random, prefix: str, count: int, *, width: int = 2
) -> list[str]:
    values: set[str] = set()
    upper = 10**width
    while len(values) < count:
        values.add(f"{prefix}{rng.randrange(upper):0{width}d}")
    return sorted(values)


_SPLIT_OFFSETS = {"train": 0, "validation": 1_000_000_007, "evaluation": 2_000_000_011}
_CAPABILITY_SURFACES: dict[str, tuple[str, ...]] = {
    "language": ("english",),
    "associative_recall": SURFACES,
    "fuzzy_recall": SURFACES,
    "selective_copying": SURFACES,
    "working_memory": SURFACES,
    "compression": SURFACES,
    "arithmetic": SURFACES,
    "symbolic": ("english", "structured", "json", "code", "symbolic"),
    "program_execution": ("english", "structured", "json", "code", "symbolic"),
    "stateful_action": ("english", "structured", "json", "code", "symbolic"),
}
_GENERATORS = {
    "language": _language,
    "associative_recall": _associative_recall,
    "fuzzy_recall": _fuzzy_recall,
    "selective_copying": _selective_copying,
    "working_memory": _working_memory,
    "compression": _compression,
    "arithmetic": _arithmetic,
    "symbolic": _symbolic,
    "program_execution": _program_execution,
    "stateful_action": _stateful_action,
}
