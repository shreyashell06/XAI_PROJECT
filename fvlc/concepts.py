"""Step 1 of Label-free CBM (concept-set generation + filtering) and the WP1 perturbation.

* `generate_concepts` queries any OpenAI-compatible chat API with the three GPT-3 prompts of
  Label-free CBM ("important features", "commonly seen around", "superclasses").
* `filter_concepts` applies the filters listed in App. C of the FVLC paper.
* `perturb_words` implements WP1: replace a fraction of the *words* of the concept set by synonyms.
  The paper asked GPT-3 to do this; by default we use WordNet (deterministic, free, seeded),
  and optionally an LLM (`method="llm"`).

By default the experiments use the concept sets released with Label-free CBM
(data/concept_sets/*_filtered.txt), which were produced by GPT-3 with exactly this procedure.
"""
import random
import re

PROMPTS = {
    "important": "List the most important features for recognizing something as a \"{}\":",
    "around": "List the things most commonly seen around a \"{}\":",
    "superclass": "Give superclasses for the word \"{}\":",
}


# --------------------------------------------------------------------------- generation
def generate_concepts(classes, model="gpt-4o-mini", prompts=("important", "around", "superclass"),
                      base_url=None, api_key=None, temperature=0.7):
    """Returns {class: [concepts]} using an OpenAI-compatible endpoint (OpenAI, Groq, Together...)."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)
    out = {}
    for c in classes:
        items = []
        for p in prompts:
            msg = PROMPTS[p].format(c) + " Answer with a short bulleted list, one item per line, no explanations."
            r = client.chat.completions.create(model=model, temperature=temperature,
                                               messages=[{"role": "user", "content": msg}])
            for line in r.choices[0].message.content.split("\n"):
                line = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip().rstrip(".")
                if line:
                    items.append(line)
        out[c] = items
    return out


def filter_concepts(concepts, classes, max_len=30, class_sim_cutoff=0.85, dup_cutoff=0.9, device="cpu"):
    """App. C filters: length, too similar to a class name, near-duplicates (sentence embeddings)."""
    from sentence_transformers import SentenceTransformer
    concepts = sorted({c.lower().strip() for c in concepts if 0 < len(c) <= max_len})
    enc = SentenceTransformer("all-mpnet-base-v2", device=device)
    ce = enc.encode(concepts, convert_to_tensor=True, normalize_embeddings=True)
    ke = enc.encode(classes, convert_to_tensor=True, normalize_embeddings=True)
    keep = (ce @ ke.T).max(dim=1).values < class_sim_cutoff
    concepts = [c for c, k in zip(concepts, keep.tolist()) if k]
    ce = ce[keep]
    sim = ce @ ce.T
    kept = []
    for i in range(len(concepts)):
        if all(sim[i, j] < dup_cutoff for j in kept):
            kept.append(i)
    return [concepts[i] for i in kept]


# --------------------------------------------------------------------------- WP1
_WORD = re.compile(r"[A-Za-z]+")


_STOP = {"with", "from", "that", "this", "have", "into", "over", "under", "their", "they", "them", "than",
          "very", "some", "many", "most", "more", "other", "such", "used", "often", "made", "like"}


def _wordnet_synonyms(word, is_head=True, n_senses=2):
    """Synonyms from the `n_senses` most frequent WordNet senses only; single lower-case words, so we
    avoid rare senses and proper nouns (e.g. black -> 'Shirley Temple Black').
    Part of speech: a word that is not the last word of its concept ("black wings") is treated as an
    adjective when WordNet has adjective senses for it; otherwise the most frequent POS is used."""
    from nltk.corpus import wordnet as wn
    w = word.lower()
    if len(w) < 4 or w in _STOP:
        return []
    senses = wn.synsets(w)
    if not senses:
        return []
    adj = [s for s in senses if s.pos() in ("a", "s")]
    if not is_head and adj:
        senses = adj
    else:
        senses = [s for s in senses if s.pos() == senses[0].pos()]
    syns = []
    for s in senses[:n_senses]:
        for l in s.lemmas():
            n = l.name()
            if ("_" in n or "-" in n or not n.islower() or n == word.lower() or n in syns
                    or n.startswith(word.lower()[:4])):
                continue
            syns.append(n)
    return syns


def _ensure_wordnet():
    """Make WordNet available *before* importing nltk: some nltk versions (e.g. 3.9.x) raise LookupError
    at `import nltk` time when WordNet is missing, so nltk.download() cannot be used to fetch it."""
    import os
    import urllib.request
    import zipfile
    root = os.environ.get("NLTK_DATA", os.path.expanduser("~/nltk_data"))
    corpora = os.path.join(root, "corpora")
    if not os.path.isdir(os.path.join(corpora, "wordnet")):
        os.makedirs(corpora, exist_ok=True)
        url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip"
        zpath = os.path.join(corpora, "wordnet.zip")
        urllib.request.urlretrieve(url, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(corpora)
    from nltk.corpus import wordnet as wn
    wn.synsets("dog")


def perturb_words(concepts, ratio, seed=0, method="wordnet", llm_model="gpt-4o-mini"):
    """WP1: replace `ratio` of all words in the concept set by a synonym.

    Returns (new_concepts, changes) where new_concepts has the same length/order as `concepts`
    (so concept index j still means the same concept) and changes is a list of
    (concept_idx, old_word, new_word).
    """
    if ratio <= 0:
        return list(concepts), []
    rng = random.Random(seed)
    positions = []
    for i, c in enumerate(concepts):
        ms = list(_WORD.finditer(c))
        positions += [(i, m.start(), m.group(), j == len(ms) - 1) for j, m in enumerate(ms)]
    n_target = max(1, round(ratio * len(positions)))

    if method == "llm":
        return _perturb_llm(concepts, ratio, llm_model)

    _ensure_wordnet()
    rng.shuffle(positions)
    chosen = []
    for i, start, w, is_head in positions:
        if len(chosen) >= n_target:
            break
        syns = _wordnet_synonyms(w, is_head)
        if syns:
            chosen.append((i, start, w, rng.choice(syns)))
    new = list(concepts)
    # replace right-to-left inside each concept so offsets stay valid
    for i, start, w, s in sorted(chosen, key=lambda t: (t[0], -t[1])):
        new[i] = new[i][:start] + s + new[i][start + len(w):]
    return new, [(i, w, s) for i, _, w, s in chosen]


def _perturb_llm(concepts, ratio, model):
    """Paper's WP1: ask the LLM to swap `ratio` of the words for synonyms, keeping one concept per line."""
    from openai import OpenAI
    client = OpenAI()
    msg = (f"Here is a list of {len(concepts)} short concepts, one per line. Replace about "
           f"{int(ratio * 100)}% of the words in the whole list with synonyms. Keep exactly one concept "
           f"per line, in the same order, and output only the list.\n\n" + "\n".join(concepts))
    r = client.chat.completions.create(model=model, temperature=0,
                                       messages=[{"role": "user", "content": msg}])
    new = [l.strip() for l in r.choices[0].message.content.split("\n") if l.strip()]
    if len(new) != len(concepts):
        raise RuntimeError("LLM changed the number of concepts; retry or use method='wordnet'.")
    changes = [(i, a, b) for i, (a, b) in enumerate(zip(concepts, new)) if a != b]
    return new, changes
