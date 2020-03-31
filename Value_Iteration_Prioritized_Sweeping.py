from itertools import product, combinations
import pickle
from random import choice, seed
from collections import defaultdict

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
    return_list = []
    for i in range(tiles(s) + 1):
        for w in combinations(values(s), i):
            if sum(w) == s[1]:
                return_list.append(tuple(1 if j + 1 in w else 0 for j in range(n_tiles)))
    return return_list


def P(a, s, s2):
    if move(s, a) != s2[0]:
        raise Exception("Action and next state not matching")
    if tile_sum(s2) > dice_sides:
        return all_dice_probabilities[s2[1]]
    else:
        if s2[1] > dice_sides:
            print(s2, "FAIL")
        return one_dice_probabilities[s2[1]]


digital_rankings = {x: n for n, x in enumerate(sorted(S, key=digital))}


def R(s, s2, scoring, scoring_func=None):
    if scoring == "jackpot":
        return 1 if tiles(s2) == 0 else 0

    if scoring == "survival":
        return tiles(s) - tiles(s2)

    if scoring == "sum":
        return tile_sum(s) - tile_sum(s2)

    if scoring == "digital":
        return digital(s) - digital(s2)

    if scoring == "digital_rank":
        return digital_rankings[s2] - digital_rankings[s]

    elif scoring_func:
        return scoring_func(s2) - scoring_func(s)


def possible_next_states(a, s):
    next_pos = move(s, a)
    return (tuple((next_pos, j)) for j in dice_range(next_pos))


def VIPS(scoring, gamma, scoring_func=None):
    sweep = [[] for i in range(n_tiles + 1)]
    for s in S:
        sweep[tiles(s)].append(s)

    V = {x: 0 for x in S}
    pi = {x: None for x in S}
    for stage in sweep:
        for s in stage:
            for a in A(s):
                v = sum(P(a, s, s2) * (R(s, s2, scoring=scoring, scoring_func=scoring_func) + gamma * V[s2]) for s2 in
                        possible_next_states(a, s))
                if v > V[s]:
                    V[s] = v
                    pi[s] = a

    return V, pi


def save_VIPS(scoring, gamma, scoring_func=None, naming_gamma=True):
    if naming_gamma:
        policy_path = f"Policies/{scoring}_g{int(gamma * 100)}.pickle"
        terminal_path = f"Terminals/{scoring}_g{int(gamma * 100)}.pickle"
    else:
        policy_path = f"Policies/{scoring}.pickle"
        terminal_path = f"Terminals/{scoring}.pickle"

    _, pi = VIPS(scoring, gamma, scoring_func)

    with open(policy_path, "wb") as file:
        pickle.dump(pi, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(terminal_path, "wb") as file:
        pickle.dump(p_terminals(pi), file, protocol=pickle.HIGHEST_PROTOCOL)


def randomized_policy(random_seed):
    seed(random_seed)
    pi = {}
    for s in S:
        actions = A(s)
        if actions:
            pi[s] = choice(actions)
        else:
            pi[s] = None

    return pi


def save_random(random_seed):
    policy_path = f"Policies/random_{random_seed}.pickle"
    terminal_path = f"Terminals/random_{random_seed}.pickle"

    pi = randomized_policy(random_seed)

    with open(policy_path, "wb") as file:
        pickle.dump(pi, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(terminal_path, "wb") as file:
        pickle.dump(p_terminals(pi), file, protocol=pickle.HIGHEST_PROTOCOL)
    


def human_policy(smart=True):
    pi = {}
    for s in S:
        actions = A(s)
        if actions:
            best = min(actions, key=lambda a: digital((a, 0))) if smart else max(actions, key=lambda a: digital((a, 0)))
            pi[s] = best
        else:
            pi[s] = None

    return pi


def save_human(smart=True):
    policy_path = f"Policies/{'anti_' if not smart else ''}human.pickle"
    terminal_path = f"Terminals/{'anti_' if not smart else ''}human.pickle"

    pi = human_policy(smart=smart)

    with open(policy_path, "wb") as file:
        pickle.dump(pi, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(terminal_path, "wb") as file:
        pickle.dump(p_terminals(pi), file, protocol=pickle.HIGHEST_PROTOCOL)


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

    else:
        raise Exception("No Scoring Function in calc_win_share")

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


def round_robin(pi_names, scoring):
    function_map = {"sum": pos_tile_sum, "digital": pos_digital, "jackpot": "jackpot"}
    n_contenders = len(pi_names)

    terminal_paths = [f"Terminals/{x}.pickle" for x in pi_names]
    terminals = [pickle_load(path) for path in terminal_paths]

    print(f"Scoring: {scoring}")
    scores = []

    for i in range(n_contenders):
        average_score = calc_average_score(terminals[i], scoring)
        scores.append(average_score)

        print(f"{pi_names[i]}: {average_score:.4f}")

    print()
    
    head_to_heads = []
    for i in range(n_contenders):
        head_to_heads.append([])
        for j in range(n_contenders):
            win_shares = calc_win_share(terminals[i], terminals[j], function_map[scoring])
            head_to_heads[-1].append(win_shares[0] + win_shares[1] / 2)
            
            print(f"{pi_names[i]}, {pi_names[j]}: {win_shares}")
    
    return scores, head_to_heads
    

def specific_scoring_func(pi_name, scoring="sum"):
    # todo invert return_func for clarity

    terminal_states = pickle_load(f"Terminals/{pi_name}.pickle")

    if scoring == "digital":
        positions_ranked = list(sorted(product((1, 0), repeat=9), key=pos_digital))

        def return_func(s):
            state_pos = s[0]
            score = sum(terminal_states[pos] for pos in positions_ranked[positions_ranked.index(state_pos) + 1:]) + \
                    terminal_states[state_pos] / 2
            return score

        return return_func

    elif scoring == "sum":
        terminals_probs = [0] * 46

        for pos, p in terminal_states.items():
            terminals_probs[pos_tile_sum(pos)] += p

        def return_func(s):
            state_sum = tile_sum(s)
            score = sum(terminals_probs[pos_sum] for pos_sum in range(state_sum + 1, 46)) + terminal_states[
                state_sum] / 2
            return score

        return return_func

    else:
        raise Exception("No scoring function in specific_scoring_func")


def directed_policy_iterating(pi_name, scoring="sum"):
    index = 1
    new_pi_name = f"{pi_name}_{scoring[0]}{index}"
    ssf = specific_scoring_func(pi_name, scoring)
    save_VIPS(new_pi_name, gamma=1, scoring_func=ssf, naming_gamma=False)

    index += 1
    old_pi_name = new_pi_name
    new_pi_name = f"{pi_name}_{scoring[0]}{index}"
    ssf = specific_scoring_func(old_pi_name, scoring)
    save_VIPS(new_pi_name, gamma=1, scoring_func=ssf, naming_gamma=False)

    while pickle_load(f"Policies/{new_pi_name}.pickle") != pickle_load(f"Policies/{old_pi_name}.pickle"):
        print(index)
        index += 1
        old_pi_name = new_pi_name
        new_pi_name = f"{pi_name}_{scoring[0]}{index}"
        ssf = specific_scoring_func(old_pi_name, scoring)
        save_VIPS(new_pi_name, gamma=1, scoring_func=ssf, naming_gamma=False)
    
    save_VIPS(f"optimal_{scoring[0]}", gamma=1, scoring_func=ssf, naming_gamma=False)

    return old_pi_name

def calc_similarity(pi_1, pi_2):
    maximum, actual = 0, 0
    for s in S:
        num_actions = len(A(s))
        if num_actions == 1:
            continue
        maximum += num_actions
        if pi_1[s] == pi_2[s]:
            actual += num_actions
    return actual/maximum

def full_comparison(pi_names):
    n_contenders = len(pi_names)

    policy_paths = [f"Policies/{x}.pickle" for x in pi_names]
    policies = [pickle_load(path) for path in policy_paths]
    
    similarities = []
    for i in range(n_contenders):
        similarities.append([])
        for j in range(n_contenders):
            similarity = calc_similarity(policies[i], policies[j])
            similarities[-1].append(similarity)
            print(f"{pi_names[i]}, {pi_names[j]}: {similarity}")
    
    return similarities