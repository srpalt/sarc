"""
Parsing + normalization pipeline for the SARC WhatsApp export CSVs.

Handles the real shape of these exports:
- one export line was independently CSV-quoted, so multi-line WhatsApp
  messages have to be re-joined by hand (header-line detection).
- a small number of rows in the raw files are corrupted: a truncated
  "[date, time] Sender:" header with no body, immediately followed by a
  duplicate of the real line. These are dropped / de-duplicated.
- WhatsApp marks every auto-generated line (system notices, media
  placeholders, call logs, deleted-message notices) with a leading
  U+200E (LEFT-TO-RIGHT MARK) character that real typed text never
  starts with. That's used as the primary system/media vs. content gate.
- real @-mentions are wrapped as "@<U+2068>Full Contact Name<U+2069>" -
  extracted separately from casual "@all"/"@everyone" text tags.
"""

import csv
import re
import bisect
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import pandas as pd

# ---------------------------------------------------------------------------
# File / chat configuration
# ---------------------------------------------------------------------------

CHAT_FILES = {
    "internal": "chat-internal.csv",
    "chair": "chat-2-chair.csv",
    "manager": "chat-3-manager.csv",
    "office": "chat-4-office.csv",
}

CHAT_LABELS = {
    "internal": "Internal Team",
    "chair": "Alumni Chair",
    "manager": "Alumni Manager",
    "office": "Alumni Office",
}

INTERNAL_CUTOFF = datetime(2026, 3, 1, 0, 0, 0)
RESPONSE_WINDOW = timedelta(minutes=30)

# ---------------------------------------------------------------------------
# Identity normalization
# ---------------------------------------------------------------------------

# Raw (invisible-char-stripped) sender name -> canonical team member name.
TEAM_MAP = {
    "Shlok Potdar": "Shlok",
    "Mansi Mishra Gim Sarc": "Mansi",
    "Debjyoti Gim BDA": "Deb",
    "Sampada Gim Sarc": "Sampada",
    "Sakshi Patil Gim Sarc": "Sakshi",
    "Khushi Dhawan Gim Sarc": "Khush Dhawan",
    "Jayesh Gim Sarc": "Jayesh",
    "Tilak Periwal Gim Sarc": "Tilak",
    "Epsha Jaiswal GIM Sarc": "Epsha",
    "+91 73790 51800": "Epsha",  # her number before she changed it on 15/04/26
    "Aditi Mehta Gim Sarc": "Aditi",
    "Ishita Vyas Gim Bda Sarc": "Ishita",
}

# Explicitly excluded per user instruction (Khushi Jain & Parth left the cell;
# Yash excluded from analytics on request).
EXCLUDED_RAW = {
    "Khushi Jain Gim Core Sarc",
    "Parth Sharma Gim Sarc JCC",
    "Yash Gaba Gim Sarc",
}

STAKEHOLDER_MAP = {
    "Sneha Ma'am Alumni Chair": "Sneha Ma'am",
    "Amit Singh GIM Alumni Manager": "Amit Sir",
    "GIM Alumni Office": "Amit Sir",  # same person, renamed contact ~Nov 2025
    "Veenita Mam Alumni Office Priya Ma'am Replacement": "Vanita Ma'am",
    "+91 99238 48960": "Priya Ma'am",  # confirmed by user
}

TEAM_ORDER = [
    "Shlok", "Mansi", "Deb", "Sampada", "Sakshi",
    "Khush Dhawan", "Jayesh", "Tilak", "Epsha", "Aditi", "Ishita",
]
STAKEHOLDER_ORDER = ["Sneha Ma'am", "Amit Sir", "Priya Ma'am", "Vanita Ma'am"]

GROUP_PSEUDO_SENDERS = {
    "SCCs\U0001f380\U0001fac2\U0001f497\U0001f60b\U0001f984\U0001fabc\U0001f419",
    "S.A.R.C. 2025-26 + Alumni Chair",
    "S.A.R.C. 2025-27 + Alumni Manager",
    "S.A.R.C. 2025-26 + Alumni Office",
    "#shlokrp",
}

# ---------------------------------------------------------------------------
# Line parsing
# ---------------------------------------------------------------------------

HEADER = re.compile(
    r"^‎?\[(\d{2})/(\d{2})/(\d{2}), (\d{2}):(\d{2}):(\d{2})\]\s(.+?):(?:\s(.*))?$"
)

SYSTEM_PATTERNS = [
    r"^Messages and calls are end-to-end encrypted",
    r"^You created the group:",
    r".+ created (this|the) group$",
    r"^You added ",
    r".+ added (you|.+)$",
    r".+ joined using a group link$",
    r".+ left$",
    r".+ removed .+$",
    r"^Your security code with .+ changed\.?$",
    r".+ changed their phone number to a new number",
    r".+ changed the group description$",
    r".+ changed the group name to",
    r".+ changed this group's icon$",
    r".+ pinned a message$",
    r"^You're now an admin$",
    r"^Disappearing messages were turned (on|off)",
    r".+ turned (on|off) disappearing messages",
    r"^(Missed (voice|video) call|Voice call|Video call),",
    r"^This message was deleted\.?$",
    r"^This message was deleted by admin",
    r"^You deleted this message\.?$",
    r"^Waiting for this message",
    r"^This message can't be displayed here",
    r"^You received a view once message",
    r"^You sent a view once message",
]
SYSTEM_RE = re.compile("|".join(f"(?:{p})" for p in SYSTEM_PATTERNS))

MEDIA_PATTERNS = [
    r"^image omitted$", r"^video omitted$", r"^audio omitted$",
    r"^sticker omitted$", r"^GIF omitted$", r"^Contact card omitted$",
    r"^POLL:",
]
MEDIA_RE = re.compile("|".join(f"(?:{p})" for p in MEDIA_PATTERNS))

MENTION_RE = re.compile(r"@[⁦⁨]([^⁩]+)⁩")
BROADCAST_RE = re.compile(r"@(all|everyone)\b", re.IGNORECASE)


def strip_invisible(s: str) -> str:
    return (
        s.replace("‎", "")
        .replace("‪", "")
        .replace("‬", "")
        .replace("⁦", "")
        .replace("⁨", "")
        .replace("⁩", "")
        .replace("\xa0", " ")
        .strip()
    )


def parse_raw_file(path: str, chat_label: str) -> list:
    """Parse one export CSV into a list of raw message dicts."""
    messages = []
    current = None
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            raw_field = row[0] if len(row) == 1 else ",".join(row)
            for line in raw_field.split("\n"):
                m = HEADER.match(line)
                if m:
                    if current:
                        messages.append(current)
                    dd, mm, yy, HH, MM, SS, sender, text = m.groups()
                    sender = strip_invisible(sender)
                    dt = datetime(2000 + int(yy), int(mm), int(dd), int(HH), int(MM), int(SS))
                    current = {
                        "chat": chat_label,
                        "datetime": dt,
                        "sender_raw": sender,
                        "text": text or "",
                    }
                else:
                    if current:
                        current["text"] += "\n" + line
    if current:
        messages.append(current)
    return messages


def _classify(messages: list) -> list:
    for msg in messages:
        raw_sender = msg["sender_raw"]
        raw_text = msg["text"]
        if raw_sender in GROUP_PSEUDO_SENDERS:
            msg["category"] = "pseudo"
            continue
        has_lrm_start = raw_text.startswith("‎")
        text = strip_invisible(raw_text)
        if not has_lrm_start:
            msg["category"] = "content"
        elif MEDIA_RE.match(text):
            msg["category"] = "media"
        elif SYSTEM_RE.match(text):
            msg["category"] = "system"
        else:
            # unmatched LRM-prefixed line: treat conservatively as content
            # (only a handful of these exist and they carry real text,
            # e.g. a message that happens to *open* with an @mention).
            msg["category"] = "content"
        msg["text_clean"] = text
    return messages


def _dedupe(messages: list) -> list:
    messages = [m for m in messages if strip_invisible(m["text"]) != ""]
    deduped = []
    prev_key = None
    for m in messages:
        key = (m["chat"], m["sender_raw"], m["datetime"], m["text"])
        if key == prev_key:
            continue
        deduped.append(m)
        prev_key = key
    return deduped


def _resolve_identity(raw_sender: str):
    """Returns (canonical_name, person_type) for a raw sender string."""
    if raw_sender in EXCLUDED_RAW:
        return None, "excluded"
    if raw_sender in TEAM_MAP:
        return TEAM_MAP[raw_sender], "team"
    if raw_sender in STAKEHOLDER_MAP:
        return STAKEHOLDER_MAP[raw_sender], "stakeholder"
    return raw_sender, "other"


def _extract_mentions(text: str):
    raw_mentions = MENTION_RE.findall(text)
    resolved = []
    for rm in raw_mentions:
        rm_clean = strip_invisible(rm)
        name, ptype = _resolve_identity(rm_clean)
        if name and ptype in ("team", "stakeholder"):
            resolved.append(name)
    is_broadcast = bool(BROADCAST_RE.search(text))
    return resolved, is_broadcast


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_messages(base_dir: str) -> pd.DataFrame:
    """Parse all four chat files into one normalized long-format DataFrame."""
    all_msgs = []
    for chat_key, fname in CHAT_FILES.items():
        path = f"{base_dir}/{fname}"
        msgs = parse_raw_file(path, chat_key)
        all_msgs.extend(msgs)

    all_msgs = _dedupe(all_msgs)
    all_msgs = _classify(all_msgs)

    rows = []
    for m in all_msgs:
        if m["category"] in ("pseudo", "system"):
            continue
        name, ptype = _resolve_identity(m["sender_raw"])
        if ptype == "excluded":
            continue
        text_clean = m.get("text_clean", strip_invisible(m["text"]))
        mentions, is_broadcast = _extract_mentions(m["text"])
        is_media = m["category"] == "media"
        word_count = 0 if is_media else len(text_clean.split())
        rows.append({
            "chat": m["chat"],
            "datetime": m["datetime"],
            "sender_raw": m["sender_raw"],
            "person": name,
            "person_type": ptype,
            "message": text_clean,
            "is_media": is_media,
            "word_count": word_count,
            "mentions": mentions,
            "is_broadcast_tag": is_broadcast,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["chat", "datetime"]).reset_index(drop=True)

    # apply internal-group cutoff
    mask_drop = (df["chat"] == "internal") & (df["datetime"] < INTERNAL_CUTOFF)
    df = df[~mask_drop].reset_index(drop=True)

    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.day_name()
    df["chat_label"] = df["chat"].map(CHAT_LABELS)

    def bucket(h):
        if 5 <= h < 12:
            return "Morning (5-12)"
        elif 12 <= h < 17:
            return "Afternoon (12-17)"
        elif 17 <= h < 22:
            return "Evening (17-22)"
        else:
            return "Late Night (22-5)"

    df["time_bucket"] = df["hour"].apply(bucket)
    return df


# ---------------------------------------------------------------------------
# Response detection (tags, @all broadcasts, stakeholder pings)
# ---------------------------------------------------------------------------

def _build_person_timelines(df: pd.DataFrame):
    """chat -> person -> sorted list of datetimes (any eligible message)."""
    timelines = {}
    for chat, g in df.groupby("chat"):
        timelines[chat] = {}
        for person, gg in g.groupby("person"):
            timelines[chat][person] = sorted(gg["datetime"].tolist())
    return timelines


def _responded_within(timelines, chat, person, after_dt, window=RESPONSE_WINDOW):
    times = timelines.get(chat, {}).get(person)
    if not times:
        return False, None
    i = bisect.bisect_right(times, after_dt)
    if i < len(times) and times[i] <= after_dt + window:
        return True, (times[i] - after_dt)
    return False, None


def compute_tag_responses(df: pd.DataFrame, window=RESPONSE_WINDOW) -> pd.DataFrame:
    """One row per (message that @-mentions someone, mentioned person)."""
    timelines = _build_person_timelines(df)
    records = []
    mentioned_rows = df[df["mentions"].map(len) > 0]
    for _, row in mentioned_rows.iterrows():
        for mentioned in set(row["mentions"]):
            if mentioned == row["person"]:
                continue
            responded, delay = _responded_within(
                timelines, row["chat"], mentioned, row["datetime"], window
            )
            records.append({
                "chat": row["chat"],
                "tagged_by": row["person"],
                "tagged_by_type": row["person_type"],
                "mentioned": mentioned,
                "datetime": row["datetime"],
                "responded": responded,
                "response_delay_sec": delay.total_seconds() if delay is not None else None,
            })
    return pd.DataFrame(records)


def compute_broadcast_responses(df: pd.DataFrame, window=RESPONSE_WINDOW) -> pd.DataFrame:
    """One row per (@all/@everyone message, team member who could have responded)."""
    timelines = _build_person_timelines(df)
    records = []
    broadcasts = df[df["is_broadcast_tag"]]
    for _, row in broadcasts.iterrows():
        for member in TEAM_ORDER:
            if member == row["person"]:
                continue
            responded, delay = _responded_within(
                timelines, row["chat"], member, row["datetime"], window
            )
            records.append({
                "chat": row["chat"],
                "broadcast_by": row["person"],
                "member": member,
                "datetime": row["datetime"],
                "responded": responded,
                "response_delay_sec": delay.total_seconds() if delay is not None else None,
            })
    return pd.DataFrame(records)


def compute_stakeholder_responses(df: pd.DataFrame, window=RESPONSE_WINDOW) -> pd.DataFrame:
    """One row per (stakeholder message in an external chat, team member)."""
    timelines = _build_person_timelines(df)
    records = []
    ext = df[(df["chat"] != "internal") & (df["person_type"] == "stakeholder")]
    for _, row in ext.iterrows():
        for member in TEAM_ORDER:
            responded, delay = _responded_within(
                timelines, row["chat"], member, row["datetime"], window
            )
            records.append({
                "chat": row["chat"],
                "stakeholder": row["person"],
                "member": member,
                "datetime": row["datetime"],
                "responded": responded,
                "response_delay_sec": delay.total_seconds() if delay is not None else None,
            })
    return pd.DataFrame(records)


def compute_external_engagement(df: pd.DataFrame, window=RESPONSE_WINDOW) -> pd.DataFrame:
    """
    For every message in an external chat (stakeholder OR teammate authored),
    per team member: did they respond within window? Lets us compare
    responsiveness-to-stakeholders vs responsiveness-to-teammates.
    """
    timelines = _build_person_timelines(df)
    records = []
    ext = df[df["chat"] != "internal"]
    for _, row in ext.iterrows():
        trigger_type = "stakeholder" if row["person_type"] == "stakeholder" else (
            "team" if row["person_type"] == "team" else "other"
        )
        if trigger_type == "other":
            continue
        for member in TEAM_ORDER:
            if member == row["person"]:
                continue
            responded, delay = _responded_within(
                timelines, row["chat"], member, row["datetime"], window
            )
            records.append({
                "chat": row["chat"],
                "trigger_person": row["person"],
                "trigger_type": trigger_type,
                "member": member,
                "datetime": row["datetime"],
                "responded": responded,
            })
    return pd.DataFrame(records)
