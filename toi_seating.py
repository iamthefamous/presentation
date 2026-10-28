"""ToiPlan: seat the guests of a toi with a Genetic Algorithm.

Rules of a good toi:
  * relatives and friends want to sit together  (+ points)
  * people with a feud must NOT share a table    (- points)
  * elders sit at the tor (table 1)              (+ points)
  * every table has exactly SEATS_PER_TABLE seats

Run:  python toi_seating.py
"""
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------- settings ----------------
SEED = 42
N_TABLES = 10
SEATS_PER_TABLE = 10
N_GUESTS = N_TABLES * SEATS_PER_TABLE
N_ELDERS = 8
N_FEUDS = 25
N_FRIENDSHIPS = 40

# happiness points
SAME_FAMILY = 10      # per pair of relatives at the same table
FRIENDS = 5           # per pair of friends at the same table
FEUD = -100           # per pair of enemies at the same table
ELDER_AT_TOR = 100    # per elder seated at the tor

# genetic algorithm
POP_SIZE = 100
GENERATIONS = 1000
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.3
ELITE = 2

RESULTS = Path(__file__).parent / "results"

MALE_NAMES = [
    "Aibek", "Akylbek", "Almaz", "Askar", "Azamat", "Bakyt", "Bektur", "Chingiz", "Daniyar", "Emil",
    "Ermek", "Islam", "Kanat", "Kubat", "Nurbek", "Nurlan", "Samat", "Talant", "Timur", "Ulan",
]
FEMALE_NAMES = [
    "Aigul", "Ainura", "Aizada", "Altynai", "Asel", "Begaiym", "Bermet", "Cholpon", "Dinara", "Elmira",
    "Gulnara", "Gulzat", "Jamila", "Kanykei", "Meerim", "Saltanat", "Tolkun", "Zamira", "Zarina", "Zhyldyz",
]
SURNAMES = [
    "Asanov", "Usenov", "Toktorov", "Mamatov", "Abdyldaev", "Sydykov", "Turdubaev",
    "Omurbekov", "Bekmuratov", "Alymkulov", "Zhumabaev", "Kasymov", "Sultanov",
    "Nurmatov", "Isakov", "Moldobaev", "Osmonov", "Satybaldiev", "Ryskulov", "Kenzhebaev",
]

rng = np.random.default_rng(SEED)


# ---------------- the guests ----------------
def make_guests():
    """Generate families, elders, feuds and friendships for one toi."""
    names, family = [], []
    f = 0
    while len(names) < N_GUESTS:
        size = min(int(rng.integers(3, 9)), N_GUESTS - len(names))
        for first in rng.choice(MALE_NAMES + FEMALE_NAMES, size, replace=False):
            surname = SURNAMES[f] + ("a" if first in FEMALE_NAMES else "")
            names.append(f"{first} {surname}")
            family.append(f)
        f += 1
    family = np.array(family)

    # the first member of some families is the elder
    first_members = [np.flatnonzero(family == k)[0] for k in range(family.max() + 1)]
    is_elder = np.zeros(N_GUESTS, dtype=bool)
    is_elder[rng.choice(first_members, N_ELDERS, replace=False)] = True

    score = np.where(family[:, None] == family[None, :], SAME_FAMILY, 0)
    feuds = random_pairs(N_FEUDS, family, score, skip=is_elder)
    for i, j in random_pairs(N_FRIENDSHIPS, family, score):
        score[i, j] = score[j, i] = FRIENDS
    for i, j in feuds:
        score[i, j] = score[j, i] = FEUD
    np.fill_diagonal(score, 0)
    return names, family, is_elder, score, feuds


def random_pairs(count, family, score, skip=None):
    """Pick `count` pairs from different families that have no relation yet."""
    pairs = []
    while len(pairs) < count:
        i, j = rng.choice(N_GUESTS, 2, replace=False)
        if family[i] == family[j] or score[i, j] != 0:
            continue
        if skip is not None and skip[i] and skip[j]:
            continue  # two elders both must sit at the tor, so they cannot feud
        score[i, j] = score[j, i] = 1  # placeholder so the pair is not picked twice
        pairs.append((i, j))
    return pairs


NAMES, FAMILY, IS_ELDER, SCORE, FEUDS = make_guests()


# ---------------- fitness ----------------
# A seating plan is a permutation of guest ids:
# seats 0-9 are table 1 (the tor), seats 10-19 are table 2, ...
def tables_of(plan):
    return plan.reshape(N_TABLES, SEATS_PER_TABLE)


def happiness(plan):
    tables = tables_of(plan)
    pairs = sum(SCORE[np.ix_(t, t)].sum() for t in tables) / 2
    return pairs + ELDER_AT_TOR * IS_ELDER[tables[0]].sum()


def describe(plan):
    """Human-readable quality numbers of a seating plan."""
    table = np.empty(N_GUESTS, dtype=int)
    table[plan] = np.repeat(np.arange(N_TABLES), SEATS_PER_TABLE)
    together = table[:, None] == table[None, :]
    upper = np.triu(np.ones_like(together), 1)
    family_pairs = (FAMILY[:, None] == FAMILY[None, :]) & upper
    friend_pairs = (SCORE == FRIENDS) & upper
    return {
        "happiness": int(happiness(plan)),
        "feuds_at_same_table": int(sum(table[i] == table[j] for i, j in FEUDS)),
        "elders_at_tor": int(IS_ELDER[tables_of(plan)[0]].sum()),
        "family_pairs_together_pct": round(float(100 * (together & family_pairs).sum() / family_pairs.sum()), 1),
        "friend_pairs_together_pct": round(float(100 * (together & friend_pairs).sum() / friend_pairs.sum()), 1),
    }


# ---------------- genetic operators ----------------
def tournament(population, scores):
    """Pick TOURNAMENT_SIZE random plans, the happiest one wins."""
    contenders = rng.choice(len(population), TOURNAMENT_SIZE, replace=False)
    return population[contenders[np.argmax(scores[contenders])]]


def crossover(mom, dad):
    """Order crossover: keep a block of mom's seats, fill the rest in dad's order."""
    a, b = sorted(rng.choice(N_GUESTS + 1, 2, replace=False))
    block = mom[a:b]
    rest = dad[~np.isin(dad, block)]
    return np.concatenate([rest[:a], block, rest[a:]])


def mutate(plan):
    """Swap two random guests."""
    i, j = rng.choice(N_GUESTS, 2, replace=False)
    plan[i], plan[j] = plan[j], plan[i]


def genetic_algorithm():
    population = [rng.permutation(N_GUESTS) for _ in range(POP_SIZE)]
    best_history, avg_history = [], []
    for _ in range(GENERATIONS):
        scores = np.array([happiness(p) for p in population])
        best_history.append(float(scores.max()))
        avg_history.append(float(scores.mean()))

        children = [population[i].copy() for i in np.argsort(scores)[-ELITE:]]
        while len(children) < POP_SIZE:
            mom, dad = tournament(population, scores), tournament(population, scores)
            child = crossover(mom, dad) if rng.random() < CROSSOVER_RATE else mom.copy()
            if rng.random() < MUTATION_RATE:
                mutate(child)
            children.append(child)
        population = children

    scores = np.array([happiness(p) for p in population])
    return population[int(np.argmax(scores))], best_history, avg_history


def random_search(tries):
    """Baseline: generate random seatings and keep the best one."""
    plans = (rng.permutation(N_GUESTS) for _ in range(tries))
    return max(plans, key=happiness)


# ---------------- output ----------------
def draw_seating(plan, title, path):
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.6))
    for t, (ax, guests) in enumerate(zip(axes.flat, tables_of(plan))):
        ax.add_patch(plt.Circle((0, 0), 1, color="#f2b705" if t == 0 else "#e8e8e8"))
        ax.text(0, 0, "Tor" if t == 0 else f"Table {t + 1}", ha="center", va="center",
                fontsize=12, fontweight="bold")
        angles = np.linspace(0, 2 * np.pi, SEATS_PER_TABLE, endpoint=False)
        pos = {g: (1.35 * np.cos(a), 1.35 * np.sin(a)) for g, a in zip(guests, angles)}
        for i, j in FEUDS:
            if i in pos and j in pos:
                ax.plot(*zip(pos[i], pos[j]), color="red", lw=2.5, zorder=1)
        for g, (x, y) in pos.items():
            ax.scatter(x, y, s=260 if IS_ELDER[g] else 130, marker="*" if IS_ELDER[g] else "o",
                       color=cmap(FAMILY[g] % 20), edgecolor="black", zorder=2)
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.7, 1.7)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.01, "colour = family    star = elder    red line = feud at the same table",
             ha="center", fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_convergence(best, avg, random_best, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(best, label="best plan", lw=2)
    ax.plot(avg, label="average plan", lw=1.5)
    ax.axhline(random_best, color="gray", ls="--",
               label=f"best of {POP_SIZE * GENERATIONS:,} random seatings")
    ax.set_xlabel("generation")
    ax.set_ylabel("happiness")
    ax.set_title("Genetic Algorithm convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_plan(plan, path):
    lines = []
    for t, guests in enumerate(tables_of(plan)):
        lines.append("Tor (table 1):" if t == 0 else f"Table {t + 1}:")
        lines += [f"  - {NAMES[g]}{' (elder)' if IS_ELDER[g] else ''}" for g in guests]
    path.write_text("\n".join(lines) + "\n")


def main():
    RESULTS.mkdir(exist_ok=True)
    seatings = math.lgamma(N_GUESTS + 1) - N_TABLES * math.lgamma(SEATS_PER_TABLE + 1)
    print(f"{N_GUESTS} guests, {FAMILY.max() + 1} families, {N_ELDERS} elders, {N_FEUDS} feuds")
    print(f"possible seatings: ~10^{seatings / math.log(10):.1f}\n")

    random_plan = rng.permutation(N_GUESTS)
    start = time.time()
    random_best = random_search(POP_SIZE * GENERATIONS)
    random_time = time.time() - start
    start = time.time()
    ga_plan, best_history, avg_history = genetic_algorithm()
    ga_time = time.time() - start

    summary = {
        "guests": N_GUESTS, "tables": N_TABLES, "families": int(FAMILY.max() + 1),
        "elders": N_ELDERS, "feuds": N_FEUDS, "friendships": N_FRIENDSHIPS,
        "log10_possible_seatings": round(seatings / math.log(10), 1),
        "random_seating": describe(random_plan),
        "best_random": {**describe(random_best), "seconds": round(random_time, 1)},
        "genetic_algorithm": {**describe(ga_plan), "seconds": round(ga_time, 1)},
        "best_history": best_history,
        "avg_history": avg_history,
    }
    for key in ("random_seating", "best_random", "genetic_algorithm"):
        print(f"{key:18} {summary[key]}")

    draw_seating(random_plan, "Random seating", RESULTS / "seating_random.png")
    draw_seating(ga_plan, "Genetic Algorithm seating", RESULTS / "seating_ga.png")
    draw_convergence(best_history, avg_history, happiness(random_best), RESULTS / "convergence.png")
    write_plan(ga_plan, RESULTS / "seating_plan.txt")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved results to {RESULTS}/")


if __name__ == "__main__":
    main()
