from itertools import product, combinations
import pickle
from collections import defaultdict
import os

n_tiles = 9
dice = 2
dice_sides = 6

all_dice_probabilities = [0] * (dice * dice_sides + 1)
for throw in product(range(1, dice_sides + 1), repeat=dice):
    all_dice_probabilities[sum(throw)] += 1 / dice_sides ** dice

one_dice_probabilities = [0.0] + [1 / dice_sides] * dice_sides


def tiles(s):
    return sum(s[0])


def values(s):
    return [i + 1 for i, x in enumerate(s[0]) if x]


def tile_sum(s):
    return sum(values(s))


def pos_tile_sum(pos):
    return tile_sum(tuple((pos, 0)))


def digital(s):
    up = values(s)
    return 0 if not up else int("".join(map(str, up)))


def pos_digital(pos):
    return digital(tuple((pos, 0)))


def move(s, a):
    return tuple(s[0][i] - a[i] for i in range(n_tiles))


def dice_range(pos):
    if sum([i + 1 for i, x in enumerate(pos) if x]) > dice_sides:
        die_range = range(dice, dice * dice_sides + 1)
    else:
        die_range = range(1, dice_sides + 1)
    return die_range


S = []
for x in product((1, 0), repeat=n_tiles):
    for die in dice_range(x):
        S.append(tuple((x, die)))

n_states = len(S)
n_positions = 2 ** n_tiles


def A(s):
    for i in range(tiles(s) + 1):
        for w in combinations(values(s), i):
            if sum(w) == s[1]:
                yield tuple(1 if j + 1 in w else 0 for j in range(n_tiles))


def P(a, s, s2):
    if move(s, a) != s2[0]:
        raise Exception("Action and next state not matching")
    if tile_sum(s2) > dice_sides:
        return all_dice_probabilities[s2[1]]
    else:
        if s2[1] > dice_sides:
            print(s2, "FAIL")
        return one_dice_probabilities[s2[1]]


digital_rankings = {x: n for n, x in enumerate(sorted(S, key=lambda s: digital(s)))}


def R(s, s2, scoring="sum"):
    if scoring == "jackpot":
        return 1 if tiles(s2) == 0 else 0

    if scoring == "survival":
        return tiles(s) - tiles(s2)

    if scoring == "sum":
        return tile_sum(s) - tile_sum(s2)

    if scoring == "digital":
        return digital(s) - digital(s2)

    if scoring == "digital_rank":
        return digital_rankings[s] - digital_rankings[s2]

    if scoring == "jackpot":
        return 1 if tile_sum(s2) == 0 else 0

    raise Exception("Non-existing scoring method")


def possible_next_states(a, s):
    next_pos = move(s, a)
    return (tuple((next_pos, j)) for j in dice_range(next_pos))


def p_terminals(pi, start_pos=tuple([1] * n_tiles)):
    policy_probabilities = defaultdict(int)

    def p_finder(position, p):
        throws = dice_range(position)
        for die in throws:
            s = tuple([position, die])

            if 1 in throws:
                p_throw = one_dice_probabilities[die]
            else:
                p_throw = all_dice_probabilities[die]

            action = pi[s]
            if action is None:
                policy_probabilities[s[0]] += p * p_throw
                continue

            p_finder(move(s, action), p * p_throw)

    p_finder(start_pos, 1)
    return policy_probabilities


def calc_average_score(pi_terminals, scoring):
    if scoring == "sum":
        pi_score = sum(pos_tile_sum(pos) * p for pos, p in pi_terminals.items())
        return pi_score

    if scoring == "digital":
        pi_score = sum(pos_digital(pos) * p for pos, p in pi_terminals.items())
        return pi_score

    if scoring == "jackpot":
        pi_score = sum(p for pos, p in pi_terminals.items() if sum(pos) == 0)
        return pi_score


def calc_win_share(pi_1_terminals, pi_2_terminals, scoring_func):
    if scoring_func == pos_digital:
        positions_ranked = list(sorted(product((1, 0), repeat=9), key=pos_digital))
        n_terminals = n_positions

        pi_1_terminals_probs = [pi_1_terminals[pos] for pos in positions_ranked]
        pi_2_terminals_probs = [pi_2_terminals[pos] for pos in positions_ranked]

    elif scoring_func == pos_tile_sum:
        n_terminals = 46

        pi_1_terminals_probs = [0] * 46
        pi_2_terminals_probs = [0] * 46

        for pos, p in pi_1_terminals.items():
            pi_1_terminals_probs[scoring_func(pos)] += p

        for pos, p in pi_2_terminals.items():
            pi_2_terminals_probs[scoring_func(pos)] += p

    pi_2_terminals_cum = [1 - pi_2_terminals_probs[0]]
    for i in range(1, n_terminals):
        pi_2_terminals_cum.append(pi_2_terminals_cum[i - 1] - pi_2_terminals_probs[i])

    pi_1_terminals_cum = [1 - pi_1_terminals_probs[0]]
    for i in range(1, n_terminals):
        pi_1_terminals_cum.append(pi_1_terminals_cum[i - 1] - pi_1_terminals_probs[i])

    pi_1_win = sum(
        pi_1_terminals_probs[i] * pi_2_terminals_cum[i] for i in range(len(pi_1_terminals_probs)))

    pi_2_win = sum(
        pi_2_terminals_probs[i] * pi_1_terminals_cum[i] for i in range(len(pi_1_terminals_probs)))

    draw = sum(pi_1_terminals_probs[i] * pi_2_terminals_probs[i] for i in range(len(pi_1_terminals_probs)))

    return pi_1_win, draw, pi_2_win


def pickle_load(path):
    with open(path, "rb") as file:
        a = pickle.load(file)
    return a


pi_names = ["sum_g100", "human"]
print(pi_names)
# policies = [pickle_load(f"Policies/{x}.pickle") for x in pi_names]
# terminals = [p_terminals(x) for x in policies]

terminal_paths = [f"Terminals/{x}.pickle" for x in pi_names]
terminals = [pickle_load(path) for path in terminal_paths]

n_contenders = len(pi_names)


def round_robin(scoring):
    function_map = {"sum": pos_tile_sum, "digital": pos_digital}

    print(f"Scoring: {scoring}")

    for i in range(n_contenders):
        print(f"{pi_names[i]}: {calc_average_score(terminals[i], scoring):.6f}")

    print()

    for i in range(n_contenders):
        for j in range(i + 1, n_contenders):
            print(
                f"{pi_names[i]}, {pi_names[j]}: {calc_win_share(terminals[i], terminals[j], function_map[scoring])}")


def save_terminals():
    for i, name in enumerate(pi_names):
        path = f"Terminals/{name}.pickle"
        with open(path, "wb") as file:
            pickle.dump(terminals[i], file, protocol=pickle.HIGHEST_PROTOCOL)


save_terminals()
round_robin("sum")
