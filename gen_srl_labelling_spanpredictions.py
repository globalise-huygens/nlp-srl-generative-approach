"""
Semantic role labelling (SRL) via the Anthropic API (default model: Claude Fable 5.1).

Input : sentences that are already tokenised AND already carry event labels
        (BIO tags such as B-TakingUnderControl), optionally entity tags too.
Output: one BIO role tag per token for each event, e.g. B-Agent, I-Time, O.

Design (adapted from the event-labelling script):
  * ONE CALL PER EVENT. A sentence can contain several events, and roles only
    make sense relative to one event. Each call marks the target event in the
    input and asks for the roles of that event only.
  * Tokens are numbered and the model returns SPANS, not one label per token:
    a list of {role, start, end, text} via a STRICT tool call
    (`submit_role_spans`; role restricted to the 11 roles). The code converts
    the spans to BIO tags. Counting 300+ per-token labels made the model lose
    count, so the output length no longer depends on the sentence length.
  * Every span is validated: role known, 0 <= start <= end < n, `text` equals
    the tokens at start..end (catches off-by-one indices), no overlap with
    other spans. Invalid output is retried; on the last attempt invalid spans
    are dropped (counted as "dropped") instead of discarding everything.
  * If all attempts fail, the event gets an all-"O" role column AND is flagged
    "failed": true in the JSON, so it can be told apart from a real "no roles".
  * Claude Fable 5.1 rejects forced tool use (tool_choice "tool"/"any"), so the
    tool is offered with tool_choice "auto" and the prompt tells the model to
    call it; a missing tool call is treated as a failed attempt and retried.
  * Fable 5.1 uses adaptive thinking (cannot be disabled) and no longer accepts
    temperature/top_p/top_k, so none are sent; max_tokens leaves room for thinking.
  * The system prompt (instructions + role glossary + few-shot examples) is a
    byte-identical prefix on every call and is marked for prompt caching.
  * Event tokens themselves are forced to "O" in the role column.
  * A role may occur several times per event (two Agents) or not at all; both
    the prompt and the BIO scheme allow this (all-"O" is a valid answer).
  * Per-event results are merged into the single role column of the TSV.
    The per-event results are also saved as JSON in case you need them.

Usage:
    python srl_labelling.py input.tsv output.tsv output.json

Input TSV (whitespace/tab separated, one token per line, blank lines ignored):
    doc  reg  anno  tok  token  event  [role]  [entity]
"""

import os
import sys
import json
import time
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple

import anthropic


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

CLIENT = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

MODEL_NAME =  "claude-opus-5-5" #  #"claude-fable-5-1"      
FALLBACK_MODEL =  "claude-sonnet-5-5" #"claude-opus-5-5"     # used if a call is refused by Fable's safety classifiers

# Adaptive thinking tokens count towards max_tokens, so leave generous headroom.
MAX_TOKENS_BASE = 8000
MAX_TOKENS_PER_TOKEN = 2               # spans are short; the output no longer scales with n

TOOL_NAME = "submit_role_spans"

USE_ENTITIES = True   # show entity tags (PER_ATTR, LOC_NAME, ...) to the model as context

LABEL_DEFINITIONS_PATH = "label_definitions_srl.md"


# ---------------------------------------------------------------------------
# 1b. Role definitions (same loader as before)
# ---------------------------------------------------------------------------

def _parse_markdown_labels(path: str) -> "OrderedDict[str, Dict]":
    labels: "OrderedDict[str, Dict]" = OrderedDict()
    current_category = None
    current_label = None
    buffer: List[str] = []

    def flush():
        if current_label is not None:
            labels[current_label] = {
                "definition": " ".join(buffer).strip(),
                "category": current_category,
            }

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("### "):
                flush()
                current_label = stripped[4:].strip()
                buffer = []
            elif stripped.startswith("## "):
                flush()
                current_label = None
                buffer = []
                current_category = stripped[3:].strip()
            elif current_label is not None and stripped:
                buffer.append(stripped)
        flush()
    return labels


def load_label_definitions(path: str) -> Dict[str, str]:
    global LABEL_CATEGORIES
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            flat = json.load(f)
        LABEL_CATEGORIES = {label: None for label in flat}
        return flat
    elif path.endswith(".md"):
        parsed = _parse_markdown_labels(path)
        LABEL_CATEGORIES = {label: info["category"] for label, info in parsed.items()}
        return {label: info["definition"] for label, info in parsed.items()}
    raise ValueError(f"Unsupported label definitions file type: {path}")


LABEL_CATEGORIES: Dict[str, Optional[str]] = {}
ROLE_DEFINITIONS: Dict[str, str] = load_label_definitions(LABEL_DEFINITIONS_PATH)
ROLES = list(ROLE_DEFINITIONS.keys())

# The actual output alphabet: O plus B-/I- for every role.
LABELS = ["O"] + [f"{p}-{r}" for r in ROLES for p in ("B", "I")]


def _format_label_glossary(definitions: Dict[str, str]) -> str:
    if not any(LABEL_CATEGORIES.values()):
        return "\n".join(f"- {label}: {d}" for label, d in definitions.items())
    grouped: "OrderedDict[Optional[str], List[str]]" = OrderedDict()
    for label, d in definitions.items():
        grouped.setdefault(LABEL_CATEGORIES.get(label), []).append(f"- {label}: {d}")
    return "\n\n".join(f"{c or 'Other'}:\n" + "\n".join(lines) for c, lines in grouped.items())


# ---------------------------------------------------------------------------
# 2. Event spans + prompt construction
# ---------------------------------------------------------------------------

Span = Tuple[int, int, str]  # (start, end inclusive, event type)


def get_event_spans(events: List[str]) -> List[Span]:
    """Turn a BIO event column into (start, end, type) spans."""
    spans: List[Span] = []
    start, etype = None, None
    for i, tag in enumerate(events):
        if tag.startswith("B-") or (tag.startswith("I-") and tag[2:] != etype):
            if start is not None:
                spans.append((start, i - 1, etype))
            start, etype = i, tag[2:]
        elif tag == "O":
            if start is not None:
                spans.append((start, i - 1, etype))
                start, etype = None, None
    if start is not None:
        spans.append((start, len(events) - 1, etype))
    return spans


SYSTEM_PROMPT = f"""You are a precise semantic role labeler for historical Dutch (VOC-era) texts.

You will receive a numbered list of tokens from a single sentence, already
split -- do NOT merge, split, reorder, add, or remove tokens. Events in the
sentence have already been identified. One event is marked <<TARGET EVENT: ...>>;
other events are marked "(other event: ...)" and are context only. Some tokens
also carry an entity tag in curly braces, e.g. {{LOC_NAME}}; use it as a hint.

Identify ONLY the semantic roles of the TARGET EVENT, as spans of tokens. Each
span has a role, a start index and an end index (both inclusive, using the
token numbers shown), and a text field with the tokens of the span, copied
exactly and joined by single spaces. Tokens that are not part of a role of the
target event are simply not listed. A span never includes the target event
tokens, and spans never overlap. Roles of other events are NOT listed.

An event can have several different roles at the same time (for example an
Agent, a Patient and a Location); list all of them that are expressed.
An event can also have several instances of the same role (for example two
Agents in a Killing event). List each instance as its own span, also when two
spans stand directly next to each other. Do not merge distinct participants
into one span.

Not every role is present for every event, and often only one role, or none,
can be recognised in the sentence. List only participants that are actually
expressed in the tokens you are given; never infer a role from background
knowledge or add one just to use more of the roles. If no role of the target
event is expressed, submit an empty list.

Roles and their definitions:

{_format_label_glossary(ROLE_DEFINITIONS)}

Follow the span-boundary conventions shown in the examples (for instance,
which function words are included in a span and which are not).

Submit your answer by calling the tool `{TOOL_NAME}` exactly once, with
"spans" set to the list of role spans (possibly empty). Do not answer in prose.
"""


def _sparse(n: int, d: Dict[int, str]) -> List[str]:
    return [d.get(i, "O") for i in range(n)]


def _make_example(tokens, events, entities, roles) -> Dict:
    n = len(tokens)
    ev = _sparse(n, events)
    return {
        "tokens": tokens,
        "events": ev,
        "entities": _sparse(n, entities),
        "roles": _sparse(n, roles),
        "target": get_event_spans(ev)[0],   # examples: first (only) event
    }


# Add more examples here, ideally including a sentence with several events and

FEW_SHOT_EXAMPLES = [
    _make_example(
        tokens=["namen", "der", "personen", "gevangen", "bij", "den", "Mataram", "t",
                "sedert", "t.", "Jaer", "1631", ".", "tot", "a", "=", "o", "1644", ".",
                "soo", "die", "gestorven", "syn", "als", "noch", "levende", ".",
                "op", "Japara", "van", "t", "Jacht", "westzanen"],
        events={3: "B-TakingUnderControl"},
        entities={2: "B-PER_ATTR", 6: "B-LOC_NAME",
                  9: "B-DATE", 10: "I-DATE", 11: "I-DATE",
                  14: "B-DATE", 15: "I-DATE", 16: "I-DATE", 17: "I-DATE",
                  28: "B-LOC_NAME", 31: "B-SHIP_TYPE", 32: "B-SHIP"},
        roles={2: "B-Patient", 6: "B-Agent",
               8: "B-Time", 9: "I-Time", 10: "I-Time", 11: "I-Time",
               13: "B-Time", 14: "I-Time", 15: "I-Time", 16: "I-Time", 17: "I-Time"},
    ),
    # --- example A: 233 tokens ---
    _make_example(
        tokens=["In", "t", "'", "Bosch", "genomen", "als", "nu", "als",
         "dan", "in", "t", "leven", "ende", "inde", "boeijen", "Jan",
         "Coertsz", "van", "Bremen", "vrijman", ".", "Jan", "van", "Lier",
         "vrijman", ".", "Joost", "Bristiaensz", "vrijman", ".", "Michiel", "Cooman",
         "van", "Uijtrecht", "soldaet", "Uijt", "de", "Boeijen", ".", "Jan",
         "Jeremiass", "vander", "Veer", "vrijman", "Dirck", "Wouterss", "van", "delffshaven",
         "Bootsman", "Rommer", "Jansz", "de", "Swaer", "van", "Willem", "Bruijn",
         "Anthonij", "Zepin", "van", "Rochel", "Jongh", "hier", "gesonden", "van",
         "Martepour", "Naervolgende", "gestorven", "ende", "gedoot", "Voor", "eerst", "drie",
         "dinaren", "van", "d", "'", "h", "=", "r", "Roelandt",
         "Crape", "Gurl", "in", "India", "van", "wegen", "sijn", "Maijs",
         "=", "t", "van", "deuemarck", "te", "weten", ".", "Pieter",
         "Croeger", "Schipper", "Jacob", "Jansz", "Vapper", "Stuurman", "Joost", "Iacobss",
         "hier", "gedoot", "om", "dat", "hij", "niet", "moors", "wilde",
         "wesen", "harmen", "Trijmerman", "vrijman", ".", "Jurian", "Tamboer", ".",
         "vrijman", ".", "lambert", "alias", "Ronsje", "vrijman", ".", "Willem",
         "pietersz", "uijt", "den", "haegh", "soldaet", "Lambert", "Lambertsz", "vrijman",
         "Johannes", "Thomass", "soldaet", "Lasel", "van", "hongarie", "hier", "comen",
         "overlopen", ".", "Jochum", "Moes", "soldaet", "Schipper", "Schodt", ".",
         "Thomas", "Lacotier", "van", "Mechelen", ",", "Soldaet", "is", "verleden",
         "Jaer", "in", "ses", "stucken", "gecapt", "doch", "om", "quaet",
         "doen", ".", "Jan", "Janss", "hier", "gebraght", "van", "Coran",
         ".", "Pieter", "Willemsz", "van", "dulkum", "soldaet", "Jan", "Willemsz",
         "van", "dulcome", "soldaet", "ontcomen", ".", "poulus", "Trompetter", "vrijman",
         "hier", "gedoot", "om", "dat", "hij", "naer", "Batavia", "geschreven",
         "hadde", ",", "en", "door", "d'onsen", "ontdect", ".", "Jeucht",
         "ende", "orangienboer", "hier", "bijde", "gedoot", ",", "om", "dat",
         "sij", "eerst", "niet", "moors", "wilden", "wesen", ",", "ende",
         "daer", "naer", "wel", "Hendrick", "Corneliss", ",", "van", "delfft",
         "soldaet"],
        events={199: "B-Communication"},
        entities={3: "B-LOC_NAME", 15: "B-PER_NAME", 16: "I-PER_NAME",
         18: "B-LOC_NAME", 19: "B-STATUS", 21: "B-PER_NAME",
         22: "I-PER_NAME", 23: "I-PER_NAME", 24: "B-STATUS",
         26: "B-PER_NAME", 27: "I-PER_NAME", 28: "B-STATUS",
         30: "B-PER_NAME", 31: "I-PER_NAME", 33: "B-LOC_NAME",
         34: "B-PRF", 39: "B-PER_NAME", 40: "I-PER_NAME",
         42: "B-LOC_NAME", 43: "B-STATUS", 44: "B-PER_NAME",
         45: "I-PER_NAME", 47: "B-LOC_NAME", 48: "B-PRF",
         49: "B-PER_NAME", 50: "I-PER_NAME", 52: "B-PER_ATTR",
         54: "B-PER_NAME", 55: "I-PER_NAME", 56: "B-PER_NAME",
         57: "I-PER_NAME", 59: "B-LOC_NAME", 60: "B-PRF",
         64: "B-LOC_NAME", 72: "B-PRF", 76: "B-PER_ATTR",
         77: "I-PER_ATTR", 78: "I-PER_ATTR", 79: "B-PER_NAME",
         80: "I-PER_NAME", 81: "B-PRF", 83: "B-LOC_NAME",
         87: "B-PRF", 88: "I-PRF", 89: "I-PRF",
         91: "B-LOC_NAME", 95: "B-PER_NAME", 96: "I-PER_NAME",
         97: "B-PRF", 98: "B-PER_NAME", 99: "I-PER_NAME",
         100: "I-PER_NAME", 101: "B-PRF", 102: "B-PER_NAME",
         103: "I-PER_NAME", 110: "B-ETH_REL", 113: "B-PER_NAME",
         114: "I-PER_NAME", 115: "B-STATUS", 117: "B-PER_NAME",
         118: "I-PER_NAME", 120: "B-STATUS", 122: "B-PER_NAME",
         124: "B-PER_NAME", 125: "B-STATUS", 127: "B-PER_NAME",
         128: "I-PER_NAME", 130: "B-LOC_NAME", 131: "I-LOC_NAME",
         132: "B-PRF", 133: "B-PER_NAME", 134: "I-PER_NAME",
         135: "B-PRF", 136: "B-PER_NAME", 137: "I-PER_NAME",
         138: "B-PRF", 139: "B-PER_NAME", 141: "B-LOC_NAME",
         146: "B-PER_NAME", 147: "I-PER_NAME", 148: "B-PRF",
         149: "B-PER_NAME", 150: "I-PER_NAME", 152: "B-PER_NAME",
         153: "I-PER_NAME", 155: "B-LOC_NAME", 157: "B-PRF",
         159: "B-DATE", 160: "I-DATE", 170: "B-PER_NAME",
         171: "I-PER_NAME", 175: "B-LOC_NAME", 177: "B-PER_NAME",
         178: "I-PER_NAME", 180: "B-LOC_NAME", 181: "B-PRF",
         182: "B-PER_NAME", 183: "I-PER_NAME", 185: "B-LOC_NAME",
         186: "B-PRF", 189: "B-PER_NAME", 190: "I-PER_NAME",
         191: "B-STATUS", 198: "B-LOC_NAME", 207: "B-PER_NAME",
         209: "B-PER_NAME", 219: "B-ETH_REL", 227: "B-PER_NAME",
         228: "I-PER_NAME", 231: "B-LOC_NAME", 232: "B-PRF"},
        roles={196: "B-Agent", 198: "B-Benefactive"},
    ),
    # --- example B: 27 tokens ---
    _make_example(
        tokens=["de", "rechte", "toenamen", ",", "ende", "met", "wat", "schepen",
         "sij", "uijtgecomen", "sijn", ",", "en", "can", "ick", "niet",
         "weten", ",", "ende", "en", "derff", "sulcx", "aen", "sommigen",
         "niet", "vragen", "."],
        events={9: "B-Translocation"},
        entities={7: "B-SHIP_TYPE"},
        roles={7: "B-Instrument", 8: "B-Patient"},
    ),
]


for _i, _ex in enumerate(FEW_SHOT_EXAMPLES):
    _bad = sorted({l for l in _ex["roles"] if l not in LABELS})
    if _bad:
        raise ValueError(f"Few-shot example {_i} uses labels not in the label set: {_bad}")


def _format_input(tokens: List[str], events: List[str],
                  entities: Optional[List[str]], target: Span) -> str:
    s, e, etype = target
    lines = []
    for i, tok in enumerate(tokens):
        notes = []
        if s <= i <= e:
            notes.append(f"<<TARGET EVENT: {etype}>>")
        elif events[i] != "O":
            notes.append(f"(other event: {events[i]})")
        if USE_ENTITIES and entities and entities[i] != "O":
            notes.append("{" + entities[i] + "}")
        lines.append(f"{i}: {tok}" + ("   " + " ".join(notes) if notes else ""))
    return "\n".join(lines)


def bio_to_spans(tokens: List[str], labels: List[str]) -> List[Dict]:
    """BIO tags -> [{"role", "start", "end", "text"}, ...] (end inclusive)."""
    spans, cur = [], None
    for i, lab in enumerate(labels):
        if lab.startswith("B-") or (lab.startswith("I-") and (cur is None or cur["role"] != lab[2:])):
            cur = {"role": lab[2:], "start": i, "end": i}
            spans.append(cur)
        elif lab.startswith("I-"):
            cur["end"] = i
        else:
            cur = None
    for sp in spans:
        sp["text"] = " ".join(tokens[sp["start"]:sp["end"] + 1])
    return spans


def _build_system_text() -> str:
    """Instructions + glossary + worked examples: one static block (cache prefix)."""
    parts = [SYSTEM_PROMPT.rstrip(), "", "# Worked examples", ""]
    for k, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        parts += [f"## Example {k}",
                  "Input:",
                  _format_input(ex["tokens"], ex["events"], ex["entities"], ex["target"]),
                  "",
                  f"Correct arguments for {TOOL_NAME}:",
                  json.dumps({"spans": bio_to_spans(ex["tokens"], ex["roles"])}, ensure_ascii=False),
                  ""]
    return "\n".join(parts)


# Built once at import time; keep it byte-for-byte identical across requests so
# the prefix can be cached (Fable's minimum cacheable prefix is ~2,000 tokens).
SYSTEM_BLOCKS = [{"type": "text", "text": _build_system_text(),
                  "cache_control": {"type": "ephemeral"}}]

ROLE_TOOL = {
    "name": TOOL_NAME,
    "description": "Submit the semantic role spans of the target event (empty list if none).",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "spans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": ROLES},
                        "start": {"type": "integer", "description": "index of the first token"},
                        "end": {"type": "integer", "description": "index of the last token (inclusive)"},
                        "text": {"type": "string", "description": "the span's tokens joined by single spaces"},
                    },
                    "required": ["role", "start", "end", "text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["spans"],
        "additionalProperties": False,
    },
}


def build_messages(tokens, events, entities, target: Span) -> List[Dict]:
    n = len(tokens)
    text = (_format_input(tokens, events, entities, target)
            + f"\n\nCall {TOOL_NAME} with the role spans of the target event "
              f"(token indices 0-{n - 1}); use an empty list if no role is expressed.")
    return [{"role": "user", "content": text}]


# ---------------------------------------------------------------------------
# 3. BIO validation / repair
# ---------------------------------------------------------------------------

def bio_errors(labels: List[str]) -> List[int]:
    """Positions where an I-X does not follow a B-X or I-X of the same role."""
    errs, prev = [], "O"
    for i, lab in enumerate(labels):
        if lab.startswith("I-") and (prev == "O" or prev[2:] != lab[2:]):
            errs.append(i)
        prev = lab
    return errs


def repair_bio(labels: List[str]) -> List[str]:
    """Turn orphan I-X tags into B-X."""
    out, prev = list(labels), "O"
    for i, lab in enumerate(out):
        if lab.startswith("I-") and (prev == "O" or prev[2:] != lab[2:]):
            out[i] = "B-" + lab[2:]
        prev = out[i]
    return out


def _squash(x: str) -> str:
    return "".join(x.split())


def spans_to_labels(tokens: List[str], spans: List[Dict], strict: bool = True) -> Tuple[List[str], int]:
    """
    Convert model spans to BIO tags. strict=True raises ValueError on the first
    invalid span (so the call can be retried); strict=False drops invalid spans
    instead. Returns (labels, number_of_dropped_spans).
    """
    n = len(tokens)
    labels, dropped = ["O"] * n, 0
    for sp in spans:
        try:
            role, st, en, text = sp["role"], sp["start"], sp["end"], sp["text"]
            if role not in ROLES:
                raise ValueError(f"unknown role {role!r}")
            if not (isinstance(st, int) and isinstance(en, int) and 0 <= st <= en < n):
                raise ValueError(f"bad indices {st}-{en} for {n} tokens")
            if _squash(text) != _squash(" ".join(tokens[st:en + 1])):
                raise ValueError(f"span {role} {st}-{en}: text {text!r} != tokens "
                                 f"{' '.join(tokens[st:en + 1])!r}")
            if any(labels[i] != "O" for i in range(st, en + 1)):
                raise ValueError(f"span {role} {st}-{en} overlaps another span")
        except (ValueError, KeyError, TypeError):
            if strict:
                raise
            dropped += 1
            continue
        labels[st] = "B-" + role
        for i in range(st + 1, en + 1):
            labels[i] = "I-" + role
    return labels, dropped


# ---------------------------------------------------------------------------
# 4. Labelling one event, then a whole sentence
# ---------------------------------------------------------------------------

def label_event(tokens: List[str], events: List[str], entities: Optional[List[str]],
                target: Span, max_retries: int = 3) -> Optional[Dict]:
    """
    Returns {"roles": BIO list, "spans": [...], "dropped": int} or None if every
    attempt failed.
    """
    n = len(tokens)
    s, e, _ = target
    model = MODEL_NAME

    for attempt in range(max_retries):
        final = attempt == max_retries - 1
        try:
            # No temperature/top_p/top_k (rejected by Fable 5.x), no forced
            # tool_choice (rejected by Fable 5.1), no thinking parameter.
            response = CLIENT.messages.create(
                model=model,
                max_tokens=MAX_TOKENS_BASE + MAX_TOKENS_PER_TOKEN * n,
                system=SYSTEM_BLOCKS,
                tools=[ROLE_TOOL],
                tool_choice={"type": "auto"},
                messages=build_messages(tokens, events, entities, target),
            )

            if response.stop_reason == "refusal":
                failed, model = model, FALLBACK_MODEL
                raise ValueError(f"{failed} refused the request; using {FALLBACK_MODEL}")
            if response.stop_reason == "max_tokens":
                raise ValueError("hit max_tokens (thinking + output); raise MAX_TOKENS_BASE")

            block = next((b for b in response.content
                          if b.type == "tool_use" and b.name == TOOL_NAME), None)
            if block is None:
                raise ValueError("model did not call the tool")

            # Strict until the last attempt; then invalid spans are dropped.
            labels, dropped = spans_to_labels(tokens, list(block.input["spans"]), strict=not final)

            # The event tokens themselves never carry a role (this can clip a span).
            for i in range(s, e + 1):
                labels[i] = "O"
            labels = repair_bio(labels)
            return {"roles": labels, "spans": bio_to_spans(tokens, labels), "dropped": dropped}

        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as err:
            print(f"[attempt {attempt + 1}] parse/validation error: {err}. Retrying...")
            time.sleep(1)
        except anthropic.BadRequestError as err:
            print(f"[attempt {attempt + 1}] request rejected (not retrying): {err}")
            return None
        except Exception as err:   # rate limits, overloads, connection errors, ...
            print(f"[attempt {attempt + 1}] API error: {err}. Retrying...")
            time.sleep(2 ** attempt)

    print(f"Failed to label event {target} after {max_retries} attempts.")
    return None


def label_sentence(tokens: List[str], events: List[str],
                   entities: Optional[List[str]] = None) -> Dict:
    """
    Returns {"events": [{"span": [s, e], "type": str, "failed": bool, "dropped": int,
                         "spans": [...], "roles": [...]}, ...],
             "merged": [...]}   (merged = single role column)
    A failed event gets an all-"O" role column and failed=True.
    """
    entities = entities or ["O"] * len(tokens)
    results = []
    for span in get_event_spans(events):
        res = label_event(tokens, events, entities, span)
        failed = res is None
        if failed:
            res = {"roles": ["O"] * len(tokens), "spans": [], "dropped": 0}
        results.append({"span": [span[0], span[1]], "type": span[2], "failed": failed, **res})
    return {"events": results, "merged": merge_role_columns(len(tokens), results)}


def merge_role_columns(n: int, results: List[Dict]) -> List[str]:
    """
    Collapse per-event role columns into one column. Where two events claim the
    same token, the earlier event wins. Overlaps are the price of a single
    column; use the per-event JSON if you need them kept apart.
    """
    merged = ["O"] * n
    for r in results:
        for i, lab in enumerate(r["roles"]):
            if merged[i] == "O" and lab != "O":
                merged[i] = lab
    return repair_bio(merged)


# ---------------------------------------------------------------------------
# 5. TSV input / output
# ---------------------------------------------------------------------------

def read_tsv(path: str) -> List[Dict]:
    """Group token lines into sentences keyed by (doc, region, annotation)."""
    sentences: "OrderedDict[Tuple[str, str, str], Dict]" = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            doc, reg, anno, tok, token, event = parts[:6]
            entity = parts[7] if len(parts) >= 8 else "O"
            key = (doc, reg, anno)
            sent = sentences.setdefault(key, {"key": key, "ids": [], "tokens": [],
                                              "events": [], "entities": []})
            sent["ids"].append(tok)
            sent["tokens"].append(token)
            sent["events"].append(event)
            sent["entities"].append(entity)
    return list(sentences.values())


def write_tsv(path: str, sentences: List[Dict], merged_roles: List[List[str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for sent, roles in zip(sentences, merged_roles):
            doc, reg, anno = sent["key"]
            for i, tok_id in enumerate(sent["ids"]):
                f.write("\t".join([doc, reg, anno, tok_id, sent["tokens"][i],
                                   sent["events"][i], roles[i], sent["entities"][i]]) + "\n")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: python srl_labelling.py input.tsv output.tsv output.json")
    in_path, out_tsv, out_json = sys.argv[1:4]

    print(f"[info] {len(ROLES)} roles, {len(LABELS)} BIO labels, loaded from {LABEL_DEFINITIONS_PATH}")
    sentences = read_tsv(in_path)
    print(f"[info] {len(sentences)} sentences, "
          f"{sum(len(get_event_spans(s['events'])) for s in sentences)} events")

    merged_all, json_all = [], []
    for sent in sentences:
        res = label_sentence(sent["tokens"], sent["events"], sent["entities"])
        merged_all.append(res["merged"])
        json_all.append({"key": list(sent["key"]), "tokens": sent["tokens"], **res})

    # Sanity statistics
    evs = [(j["key"], ev) for j in json_all for ev in j["events"]]
    failed = [(k, ev) for k, ev in evs if ev["failed"]]
    empty = sum(1 for _, ev in evs if not ev["failed"] and not ev["spans"])
    dropped = sum(ev["dropped"] for _, ev in evs)
    multi = {}
    for _, ev in evs:
        for role in ROLES:
            if sum(1 for sp in ev["spans"] if sp["role"] == role) > 1:
                multi[role] = multi.get(role, 0) + 1
    print(f"[stats] {len(evs)} events: {len(failed)} FAILED (written as all-O, flagged in JSON), "
          f"{empty} without any role, {dropped} invalid spans dropped; "
          f"events with >1 span of the same role: {multi or 'none'}")
    for k, ev in failed:
        print(f"  failed: {':'.join(k)} event {ev['type']} at token {ev['span'][0]}")

    write_tsv(out_tsv, sentences, merged_all)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_all, f, ensure_ascii=False, indent=1)
    print(f"[done] wrote {out_tsv} and {out_json}")
