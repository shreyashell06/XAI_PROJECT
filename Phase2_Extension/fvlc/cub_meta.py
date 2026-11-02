"""CUB-200-2011 human annotations and the mapping from LLM concepts to them (Phase 2).

Files are read from data/CUB_meta/ (written by scripts/download_cub.sh):
  images.txt                  <image_id> <class_dir>/<file>.jpg
  bounding_boxes.txt          <image_id> <x> <y> <w> <h>
  parts.txt                   <part_id> <part name>                 (15 parts)
  part_locs.txt               <image_id> <part_id> <x> <y> <visible>
  attributes.txt              <attr_id> has_<part>_color::<colour> (312 attributes)
  image_attribute_labels.txt  <image_id> <attr_id> <is_present> <certainty_id> <time>
                              certainty: 1 not visible, 2 guessing, 3 probably, 4 definitely

The concepts are free text written by an LLM ("a black back and white belly"), so we map them with a small,
explicit word list. Two mappings are produced:
  * keypoint parts  -> used by part-masking locality (which body parts carry the concept's evidence)
  * (part, colour)  -> used by attribute AUROC (which human attribute labels say the concept is present)
Every concept that cannot be mapped is simply left out of the corresponding metric.
"""
import os
import re
from dataclasses import dataclass, field

import torch

META_ROOT = os.environ.get("FVLC_CUB_META", "data/CUB_meta")

# word -> (keypoint part names, CUB attribute part names). Matching is on whole lower-case tokens; a token
# matches a key if it equals it or starts with it and the key is marked as a prefix ("wing*" covers wings,
# wingtip, wingtips).
PART_WORDS = {
    "wing*": (["left wing", "right wing"], ["wing"]),
    "beak*": (["beak"], ["bill"]),
    "bill": (["beak"], ["bill"]),
    "bills": (["beak"], ["bill"]),
    "tail*": (["tail"], ["upper_tail", "under_tail"]),
    "crown": (["crown"], ["crown"]),
    "cap": (["crown"], ["crown"]),
    "forehead": (["forehead"], ["forehead"]),
    "head": (["crown", "forehead"], ["crown", "forehead"]),
    "heads": (["crown", "forehead"], ["crown", "forehead"]),
    "breast*": (["breast"], ["breast"]),
    "chest": (["breast"], ["breast"]),
    "belly": (["belly"], ["belly"]),
    "underbelly": (["belly"], ["belly"]),
    "underparts": (["breast", "belly"], ["underparts"]),
    "underside": (["breast", "belly"], ["underparts"]),
    "back": (["back"], ["back"]),
    "mantle": (["back"], ["back"]),
    "upperparts": (["back"], ["upperparts"]),
    "leg*": (["left leg", "right leg"], ["leg"]),
    "feet": (["left leg", "right leg"], ["leg"]),
    "foot": (["left leg", "right leg"], ["leg"]),
    "eye": (["left eye", "right eye"], ["eye"]),
    "eyes": (["left eye", "right eye"], ["eye"]),
    "eyeline": (["left eye", "right eye"], []),     # a stripe, not the eye colour
    "eyebrows": (["left eye", "right eye"], []),
    "eyering": (["left eye", "right eye"], []),
    "throat": (["throat"], ["throat"]),
    "chin": (["throat"], ["throat"]),
    "nape": (["nape"], ["nape"]),
    "neck": (["nape"], ["nape"]),
    "collar": (["nape"], ["nape"]),
}

# colour word -> CUB colour name
COLOUR_WORDS = {
    "blue": "blue", "bluish": "blue", "brown": "brown", "brownish": "brown", "iridescent": "iridescent",
    "purple": "purple", "violet": "purple", "rufous": "rufous", "rusty": "rufous", "rust": "rufous",
    "chestnut": "rufous", "grey": "grey", "gray": "grey", "greyish": "grey", "grayish": "grey",
    "yellow": "yellow", "yellowish": "yellow", "golden": "yellow", "olive": "olive", "green": "green",
    "greenish": "green", "pink": "pink", "pinkish": "pink", "rosy": "pink", "orange": "orange",
    "orangish": "orange", "black": "black", "blackish": "black", "white": "white", "whitish": "white",
    "red": "red", "reddish": "red", "scarlet": "red", "crimson": "red", "buff": "buff", "tan": "buff",
    "cream": "buff",
}

# Nouns for the whole bird: a colour attached to them is not a part colour.
BODY_NOUNS = {"body", "bodies", "plumage", "bird", "birds", "feathers", "coloration", "colour", "color", "scheme",
              "sheen", "overall", "upper", "lower"}

# Canonical masking groups for locality: each is a set of keypoints masked together.
CANONICAL_GROUPS = {
    "wing": ("left wing", "right wing"), "beak": ("beak",), "tail": ("tail",), "head": ("crown", "forehead"),
    "breast": ("breast",), "belly": ("belly",), "back": ("back",), "leg": ("left leg", "right leg"),
    "eye": ("left eye", "right eye"), "throat": ("throat",), "nape": ("nape",),
}


def _part_entry(token):
    if token in PART_WORDS:
        return PART_WORDS[token]
    for key, val in PART_WORDS.items():
        if key.endswith("*") and token.startswith(key[:-1]):
            return val
    return None


@dataclass
class ConceptMap:
    concept: str
    kp_parts: tuple = ()                            # keypoint part names (for locality)
    groups: list = field(default_factory=list)      # [(attr_part_alternatives, colours)]: AND across entries,
                                                    # OR across the alternatives x colours of one entry


def parse_concept(concept: str) -> ConceptMap:
    """Map one concept to keypoint parts and (part, colour) groups.

    Colours attach to the part nouns that follow them ("a brown back and white belly" -> brown back AND white
    belly). Parts in one phrase share its colours and must all match ("black wings and tail" -> black wing AND
    black tail). Colours after the last part apply to the preceding colourless parts ("beak and legs are
    black"). "head" may match the crown or the forehead colour.
    """
    tokens = re.findall(r"[a-z]+", concept.lower())
    groups = [[[], []]]  # [[attr-part alternatives per part word], colours]
    kp = []
    for t in tokens:
        if t in COLOUR_WORDS:
            if groups[-1][0]:            # previous noun phrase is closed -> new group
                groups.append([[], []])
            if COLOUR_WORDS[t] not in groups[-1][1]:
                groups[-1][1].append(COLOUR_WORDS[t])
            continue
        if t in BODY_NOUNS and groups[-1][1] and not groups[-1][0]:
            groups[-1][1] = []           # "a yellow body with black wings": yellow describes the body, drop it
            continue
        entry = _part_entry(t)
        if entry is None:
            continue
        kparts, aparts = entry
        kp += [p for p in kparts if p not in kp]
        groups[-1][0].append(tuple(aparts))   # empty tuple = a part with no colour attribute (e.g. eyeline)
    # trailing colours with no part -> give them to the previous group if it has none
    if len(groups) > 1 and not groups[-1][0] and groups[-1][1] and not groups[-2][1]:
        groups[-2][1] = groups[-1][1]
        groups.pop()
    clean = []
    for words, cols in groups:
        if not words and not cols:
            continue
        if not words or not cols or any(len(w) == 0 for w in words):
            clean = None             # colour without part, part without colour, or a part with no attribute
            break
        clean += [(w, tuple(cols)) for w in dict.fromkeys(words)]
    return ConceptMap(concept, tuple(kp), clean or [])


def map_concepts(concepts):
    return [parse_concept(c) for c in concepts]


def _read(name):
    path = os.path.join(META_ROOT, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Re-run `bash scripts/download_cub.sh` (Phase 2 version).")
    with open(path) as f:
        return [l.split() for l in f if l.strip()]


class CUBMeta:
    """All annotations, indexed by CUB image id (int)."""

    def __init__(self, root=None):
        global META_ROOT
        if root:
            META_ROOT = root
        self.path_of = {int(i): p for i, p in _read("images.txt")}
        self.id_of = {p: i for i, p in self.path_of.items()}
        self.bbox = {int(r[0]): tuple(map(float, r[1:5])) for r in _read("bounding_boxes.txt")}
        self.part_names = {int(r[0]): " ".join(r[1:]) for r in _read("parts.txt")}
        self.part_id = {n: i for i, n in self.part_names.items()}
        self.kp = {}  # image_id -> (15, 3) tensor: x, y, visible (part ids 1..15 -> rows 0..14)
        for r in _read("part_locs.txt"):
            i, p = int(r[0]), int(r[1])
            t = self.kp.setdefault(i, torch.zeros(len(self.part_names), 3))
            t[p - 1] = torch.tensor([float(r[2]), float(r[3]), float(r[4])])
        self.attr_names = {int(r[0]): r[1] for r in _read("attributes.txt")}
        self.attr_id = {n: i for i, n in self.attr_names.items()}
        self._labels = None

    def attribute_labels(self):
        """(present, certainty) as dicts image_id -> (312,) tensors. Lines with extra columns are tolerated."""
        if self._labels is None:
            n = max(self.attr_names)
            present, cert = {}, {}
            for r in _read("image_attribute_labels.txt"):
                i, a = int(r[0]), int(r[1])
                present.setdefault(i, torch.zeros(n + 1))[a] = float(r[2])
                cert.setdefault(i, torch.zeros(n + 1))[a] = float(r[3])
            self._labels = (present, cert)
        return self._labels

    def image_ids(self, samples):
        """ImageFolder `samples` (path, class) -> CUB image ids, matched on '<class_dir>/<file>'."""
        out = []
        for path, _ in samples:
            rel = "/".join(path.replace("\\", "/").split("/")[-2:])
            out.append(self.id_of[rel])
        return out

    def attr_ids_for(self, attr_part, colour):
        name = f"has_{attr_part}_color::{colour}"
        return [self.attr_id[name]] if name in self.attr_id else []

    def concept_labels(self, cmap: ConceptMap, image_ids, min_certainty=3):
        """Per-image label for a concept: 1 if every (part, colour) entry is present, 0 if not, -1 if the
        annotation is too uncertain (no attribute of some entry annotated with certainty >= min_certainty)."""
        present, cert = self.attribute_labels()
        labels = torch.full((len(image_ids),), -1.0)
        if not cmap.groups:
            return labels
        for n, i in enumerate(image_ids):
            if i not in present:
                continue
            ok, val = True, 1.0
            for aparts, cols in cmap.groups:
                ids = [a for p in aparts for c in cols for a in self.attr_ids_for(p, c)]
                if not ids:
                    ok = False
                    break
                if max(cert[i][a].item() for a in ids) < min_certainty:
                    ok = False
                    break
                if not any(present[i][a].item() > 0 and cert[i][a].item() >= min_certainty for a in ids):
                    val = 0.0
            if ok:
                labels[n] = val
        return labels
