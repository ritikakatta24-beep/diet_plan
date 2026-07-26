"""
BodySync - 7-Day Diet Plan Generator API
------------------------------------------
Input  : (a) total_calories, protein_g, carbs_g, fats_g — these come from the
             macro-nutrient prediction API's output.
         (b) diet_type ("veg" / "non-veg") — this does NOT come from the
             macro-nutrient API (its training data has no diet_type column).
             This is a user profile preference — same category as age/gender —
             collect it directly from the user (e.g. an onboarding toggle) and
             pass it straight through to this API, independent of any model call.
Output : a 7-day diet plan, each day broken into breakfast/lunch/dinner/snack,
         with exact dish quantities (roti count, bowl fractions, etc.)

Pipeline:
  1. split_daily_to_meals()   -> per-meal macro targets (nutrition-science rules)
  2. select_dish_combo()      -> pick WHICH dishes go in a meal (rule-based, uses component_type)
  3. optimize_portions()      -> figure out HOW MUCH of each dish (scipy optimizer)
  4. generate_day_plan()      -> combine all meals for one day
  5. generate_week_plan()     -> loop 7 days with shuffle/no-repeat tracking

"""

import os
import random
import itertools
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from flask import Flask, request, jsonify
import traceback

# The dish CSV always ships in the same folder as this script, so we build
# its path relative to this file's own location. This works no matter which
# directory the app is launched from (local run, gunicorn, Procfile, etc.) —
# unlike a bare filename, which only works if you happen to run from this
# exact folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISH_CSV_PATH = os.path.join(BASE_DIR, "bodysync_dish_dataset-3.csv")

# ---------------------------------------------------------------------------
# LAYER 1: split one day's total macros into per-meal targets
# ---------------------------------------------------------------------------

# calorie share per meal (must sum to 1.0)
CALORIE_SPLIT = {"breakfast": 0.25, "lunch": 0.30, "dinner": 0.28, "snack": 0.17}

# carbs are front-loaded (more early in the day), fats stay roughly calorie-proportional,
# protein is spread EVENLY across meals (better for muscle protein synthesis) rather than
# following the calorie split.
CARB_SPLIT = {"breakfast": 0.30, "lunch": 0.35, "dinner": 0.20, "snack": 0.15}
FAT_SPLIT = {"breakfast": 0.25, "lunch": 0.30, "dinner": 0.28, "snack": 0.17}


def split_daily_to_meals(total_calories, protein_g, carbs_g, fats_g):
    """Return dict: meal_name -> {calories, protein_g, carbs_g, fats_g}"""
    meals = {}
    n_meals = len(CALORIE_SPLIT)
    for meal in CALORIE_SPLIT:
        # Protein is meant to be spread fairly evenly across meals (nutrition-science
        # principle), BUT a purely even split breaks down for low-calorie meals like
        # snacks — asking a 374-cal snack to supply 25% of a high total protein target
        # can imply an impossible protein-density (>45% of that meal's calories).
        # Blend even-split with calorie-proportional split so smaller meals aren't
        # assigned protein they physically can't carry within realistic portions.
        protein_share = 0.55 * (1 / n_meals) + 0.45 * CALORIE_SPLIT[meal]
        meals[meal] = {
            "calories": round(total_calories * CALORIE_SPLIT[meal], 1),
            "protein_g": round(protein_g * protein_share, 1),
            "carbs_g": round(carbs_g * CARB_SPLIT[meal], 1),
            "fats_g": round(fats_g * FAT_SPLIT[meal], 1),
        }
    return meals


# ---------------------------------------------------------------------------
# LAYER 2: pick WHICH dishes go into a meal (exhaustive search, not random)
# ---------------------------------------------------------------------------
#
# Why exhaustive instead of random sampling: the old approach randomly generated
# ~20 candidate combos and kept the best. For lunch/dinner (grain x curry x side)
# that's ~8000 possible combos — 20 random samples covers ~0.25% of the space.
# That's why results were inconsistent run-to-run AND missed good combos for
# extreme (weight-loss/high-protein) targets. Since these pool sizes are small
# enough (low thousands), we can score EVERY combo with vectorized numpy in
# milliseconds, and deterministically pick from the genuinely best-fitting ones.

def _ratio_score(protein_ratio, carb_ratio, fat_ratio, target_p, target_c, target_f, bonus=0.0):
    return (
        2.0 * (protein_ratio - target_p) ** 2
        + 2.3 * (fat_ratio - target_f) ** 2
        + 1.0 * (carb_ratio - target_c) ** 2
        + bonus
    )


def select_dish_combo(dishes_df, meal_type, diet_type, target, exclude_names=None, top_k=5):
    """
    Exhaustively scores ALL valid dish combos for this meal slot against the
    target's macro RATIO (protein/carb/fat as % of calories — quantity-independent),
    then returns one randomly chosen from the top_k best-fitting combos (for
    day-to-day variety without sacrificing quality).
    """
    exclude_names = exclude_names or set()
    pool = dishes_df[dishes_df["meal_type"] == meal_type]
    if diet_type == "veg":
        pool = pool[pool["diet_type"] == "veg"]

    def get_pool(component_type):
        base = pool[pool["component_type"] == component_type]
        # Protein-dense sides/shakes/snacks (e.g. Whey Protein Shake, Paneer Bhurji
        # Side) are realistically eaten daily by people chasing a protein target —
        # unlike mains, they don't need day-to-day variety. Excluding them via the
        # no-repeat window was starving high-protein days of the only dishes that
        # could actually hit the target, forcing a fallback to carb-heavy mains on
        # 5+ of 7 days. So: exempt any dish with protein_density > 0.35 from the
        # recently-used exclusion; keep the exclusion for everything else.
        is_protein_dense = (base["base_protein_g"] * 4 / base["base_calories"].replace(0, np.nan)) > 0.35
        exempt = base[is_protein_dense.fillna(False)]
        rest = base[~is_protein_dense.fillna(False)]
        sub = pd.concat([exempt, rest[~rest["dish_name"].isin(exclude_names)]])
        if len(sub) == 0:
            sub = base
        return sub.to_dict("records")

    cal = target["calories"]
    if cal <= 0:
        return []
    target_p = (target["protein_g"] * 4) / cal
    target_c = (target["carbs_g"] * 4) / cal
    target_f = (target["fats_g"] * 9) / cal

    candidates = []  # list of (score, [dish_dict, ...])

    if meal_type == "lunch_dinner":
        grains = get_pool("grain")
        curries = get_pool("curry")
        sides = get_pool("side") + [None]  # side is optional

        if grains and curries:
            def arr(items, key):
                return np.array([it[key] if it else 0.0 for it in items], dtype=float)

            g_cal, g_p, g_c, g_f = arr(grains, "base_calories"), arr(grains, "base_protein_g"), arr(grains, "base_carbs_g"), arr(grains, "base_fats_g")
            c_cal, c_p, c_c, c_f = arr(curries, "base_calories"), arr(curries, "base_protein_g"), arr(curries, "base_carbs_g"), arr(curries, "base_fats_g")
            s_cal, s_p, s_c, s_f = arr(sides, "base_calories"), arr(sides, "base_protein_g"), arr(sides, "base_carbs_g"), arr(sides, "base_fats_g")

            tot_cal = g_cal[:, None, None] + c_cal[None, :, None] + s_cal[None, None, :]
            tot_p = g_p[:, None, None] + c_p[None, :, None] + s_p[None, None, :]
            tot_c = g_c[:, None, None] + c_c[None, :, None] + s_c[None, None, :]
            tot_f = g_f[:, None, None] + c_f[None, :, None] + s_f[None, None, :]

            safe_cal = np.clip(tot_cal, 1, None)
            pr, cr, fr = (tot_p * 4) / safe_cal, (tot_c * 4) / safe_cal, (tot_f * 9) / safe_cal
            scores = _ratio_score(pr, cr, fr, target_p, target_c, target_f)

            flat_order = np.argsort(scores, axis=None)[: top_k * 4]
            gi_arr, ci_arr, si_arr = np.unravel_index(flat_order, scores.shape)
            for gi, ci, si in zip(gi_arr, ci_arr, si_arr):
                combo = [grains[gi], curries[ci]]
                if sides[si] is not None:
                    combo.append(sides[si])
                candidates.append((float(scores[gi, ci, si]), combo))

        # light single-dish structure (e.g. Poha/Upma) — only meaningfully favored
        # when the target itself is genuinely low-calorie (weight-loss dinner etc.)
        light_bonus = -0.05 if cal < 450 else 0.05
        for d in get_pool("complete"):
            dcal = d["base_calories"]
            if dcal <= 0:
                continue
            pr, cr, fr = (d["base_protein_g"] * 4) / dcal, (d["base_carbs_g"] * 4) / dcal, (d["base_fats_g"] * 9) / dcal
            score = _ratio_score(pr, cr, fr, target_p, target_c, target_f, bonus=light_bonus)
            candidates.append((score, [d]))

    elif meal_type == "breakfast":
        mains = get_pool("complete")
        sides = get_pool("side") + [None]
        for m in mains:
            for s in sides:
                dcal = m["base_calories"] + (s["base_calories"] if s else 0)
                if dcal <= 0:
                    continue
                dp = m["base_protein_g"] + (s["base_protein_g"] if s else 0)
                dc = m["base_carbs_g"] + (s["base_carbs_g"] if s else 0)
                df = m["base_fats_g"] + (s["base_fats_g"] if s else 0)
                pr, cr, fr = (dp * 4) / dcal, (dc * 4) / dcal, (df * 9) / dcal
                score = _ratio_score(pr, cr, fr, target_p, target_c, target_f)
                candidates.append((score, [m] + ([s] if s else [])))

    elif meal_type == "snack":
        items = get_pool("snack") + get_pool("complete") + get_pool("beverage")
        for d in items:
            dcal = d["base_calories"]
            if dcal <= 0:
                continue
            pr, cr, fr = (d["base_protein_g"] * 4) / dcal, (d["base_carbs_g"] * 4) / dcal, (d["base_fats_g"] * 9) / dcal
            score = _ratio_score(pr, cr, fr, target_p, target_c, target_f)
            candidates.append((score, [d]))
        for d1, d2 in itertools.combinations(items, 2):
            dcal = d1["base_calories"] + d2["base_calories"]
            if dcal <= 0:
                continue
            dp = d1["base_protein_g"] + d2["base_protein_g"]
            dc = d1["base_carbs_g"] + d2["base_carbs_g"]
            df = d1["base_fats_g"] + d2["base_fats_g"]
            pr, cr, fr = (dp * 4) / dcal, (dc * 4) / dcal, (df * 9) / dcal
            score = _ratio_score(pr, cr, fr, target_p, target_c, target_f)
            candidates.append((score, [d1, d2]))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0])
    # Cheap ratio-based prefilter already narrowed this to a reasonable shortlist
    # (top_k*4 for lunch/dinner, or all candidates for smaller meal types). Now
    # re-rank that shortlist by ACTUAL achievability: run the real portion
    # optimizer on each and score by its true post-optimization error, since a
    # combo's 1x ratio can look fine while being infeasible to hit exactly once
    # quantities are scaled (e.g. a fatty protein dish needing to be scaled up
    # for protein, dragging fat over target along with it).
    rerank_pool = candidates[: min(len(candidates), 6)]
    rescored = []
    for _, combo in rerank_pool:
        _, totals = optimize_portions(combo, target)
        real_err = sum(ERROR_WEIGHTS[k] * (totals[k] - target[k]) ** 2 for k in ERROR_WEIGHTS)
        rescored.append((real_err, combo))
    rescored.sort(key=lambda x: x[0])
    top = rescored[:top_k]
    _, chosen_combo = random.choice(top)
    return chosen_combo


# ---------------------------------------------------------------------------
# LAYER 3: figure out HOW MUCH of each dish (scipy optimizer + rounding)
# ---------------------------------------------------------------------------

# weights: calories and protein matter most, carbs/fats can flex a bit more
ERROR_WEIGHTS = {"calories": 1.5, "protein_g": 1.8, "carbs_g": 1.0, "fats_g": 2.1}


def _macro_error(qty_vector, dishes, target):
    total = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fats_g": 0.0}
    for qty, dish in zip(qty_vector, dishes):
        total["calories"] += qty * dish["base_calories"]
        total["protein_g"] += qty * dish["base_protein_g"]
        total["carbs_g"] += qty * dish["base_carbs_g"]
        total["fats_g"] += qty * dish["base_fats_g"]

    err = 0.0
    for key, w in ERROR_WEIGHTS.items():
        diff = total[key] - target[key]
        err += w * (diff ** 2)
    return err


def _get_adaptive_bounds(target):
    """
    Default bounds (0.5x-1.75x) work fine for 'normal/maintenance' macro ratios,
    but are too narrow when the target is a high-protein, very-low-fat, OR
    high-carb profile — the optimizer needs more room to scale the relevant
    dishes (protein-dense, fat-light, or carb-dense/grain) up or down to hit
    those targets. Without this, quantity gets clipped at the bound and the
    macro is silently under-delivered even though a better-fitting quantity
    exists mathematically (this was measured directly: grain/carb dishes were
    landing exactly on the 1.75x cap for high-carb targets).
    """
    if target["calories"] <= 0:
        return (0.5, 1.75)
    protein_ratio = (target["protein_g"] * 4) / target["calories"]
    fat_ratio = (target["fats_g"] * 9) / target["calories"]
    carb_ratio = (target["carbs_g"] * 4) / target["calories"]

    is_extreme = (
        protein_ratio > 0.24 or fat_ratio < 0.18 or fat_ratio > 0.32 or carb_ratio > 0.58
    )
    if is_extreme:
        return (0.3, 2.5)
    return (0.5, 1.75)


def optimize_portions(dishes, target, tolerance_pct=10):
    """
    dishes: list of dish dicts (from select_dish_combo)
    target: {"calories":.., "protein_g":.., "carbs_g":.., "fats_g":..}
    Returns: list of {dish_name, quantity, unit_type, calories, protein_g, carbs_g, fats_g}
             + the achieved meal totals
    """
    if not dishes:
        return [], {"calories": 0, "protein_g": 0, "carbs_g": 0, "fats_g": 0}

    n = len(dishes)
    bound_range = _get_adaptive_bounds(target)
    bounds = [bound_range for _ in range(n)]

    # Macro error function is convex quadratic, so single L-BFGS-B run from x0=1.0
    # reliably converges to the global minimum in milliseconds.
    best_result = minimize(
        _macro_error, np.ones(n), args=(dishes, target),
        bounds=bounds, method="L-BFGS-B"
    )
    raw_qty = best_result.x

    # --- rounding pass: snap each dish to its allowed increment ---
    rounded_qty = []
    for qty, dish in zip(raw_qty, dishes):
        increment = float(dish["portion_increment"])
        snapped = round(qty / increment) * increment
        if dish["unit_type"] == "discrete":
            snapped = max(1, round(snapped))  # never 0 rotis, always whole number
        rounded_qty.append(snapped)

    # --- rebalance pass: nudge CONTINUOUS dishes to compensate for rounding error ---
    continuous_idx = [i for i, d in enumerate(dishes) if d["unit_type"] == "continuous"]
    if continuous_idx:
        def rebalance_error(cont_qty_vector):
            full_qty = list(rounded_qty)
            for idx, val in zip(continuous_idx, cont_qty_vector):
                full_qty[idx] = val
            return _macro_error(full_qty, dishes, target)

        cont_bounds = [(0.25, bound_range[1]) for _ in continuous_idx]
        cont_x0 = [rounded_qty[i] for i in continuous_idx]
        cont_result = minimize(rebalance_error, cont_x0, bounds=cont_bounds, method="L-BFGS-B")
        for idx, val in zip(continuous_idx, cont_result.x):
            increment = float(dishes[idx]["portion_increment"])
            rounded_qty[idx] = round(val / increment) * increment

    # --- build final output ---
    final_dishes = []
    totals = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fats_g": 0.0}
    for qty, dish in zip(rounded_qty, dishes):
        cal = round(qty * dish["base_calories"], 1)
        pro = round(qty * dish["base_protein_g"], 1)
        carb = round(qty * dish["base_carbs_g"], 1)
        fat = round(qty * dish["base_fats_g"], 1)
        totals["calories"] += cal
        totals["protein_g"] += pro
        totals["carbs_g"] += carb
        totals["fats_g"] += fat
        final_dishes.append({
            "dish_name": dish["dish_name"],
            "quantity": qty,
            "base_portion": dish["base_portion"],
            "unit_type": dish["unit_type"],
            "calories": cal, "protein_g": pro, "carbs_g": carb, "fats_g": fat,
        })

    totals = {k: round(v, 1) for k, v in totals.items()}
    return final_dishes, totals


def _within_tolerance(totals, target, tolerance_pct=10):
    for key in ["calories", "protein_g", "carbs_g", "fats_g"]:
        t = target[key]
        if t == 0:
            continue
        pct_diff = abs(totals[key] - t) / t * 100
        if pct_diff > tolerance_pct:
            return False
    return True


# ---------------------------------------------------------------------------
# LAYER 3.5: post-hoc check — did we actually hit protein/fat, not just the
# static pre-flight ratio guess? This is the more trustworthy signal because
# it reflects what the optimizer/dish-pool actually produced, not a heuristic
# estimate of what "should" be achievable.
#
# NOTE: this is still computed per-day (kept for internal telemetry / possible
# future use), but the resulting message is no longer attached to the API
# response — see generate_day_plan() / generate_week_plan() below.
# ---------------------------------------------------------------------------

SUPPLEMENT_TRIGGER_PCT = 10  # deviation above which we'd otherwise surface a message

# Concrete, actionable suggestions when a macro is SHORT of target.
SUPPLEMENT_OPTIONS = {
    "protein": "a protein shake, paneer, or Greek yogurt",
    "fat": "a spoon of peanut butter, nuts, or ghee",
}

# What to say when a macro OVERSHOOTS target — there's no single food to "add"
# for this, so the message explains the likely cause instead of suggesting a fix.
OVERSHOOT_EXPLANATIONS = {
    "protein": "the available high-protein dishes for this diet type also carry more of the other macros than the target allows",
    "carbs": "vegetarian mains are largely grain-based, so carbs are hard to bring down further at this protein/calorie level",
    "fat": "high-protein vegetarian sources (paneer, soya) carry meaningful fat, which is hard to avoid while still hitting the protein target",
}


def _check_supplement_need(day_totals, daily_targets):
    """
    Returns (needs_attention: bool, message: str|None, deviations: dict).

    Kept for internal use only — callers in this file no longer attach the
    message/flag to the API response.
    """
    shortfalls = {}
    overshoots = {}
    for key, label in (("protein_g", "protein"), ("carbs_g", "carbs"), ("fats_g", "fat")):
        target = daily_targets[key]
        if target <= 0:
            continue
        actual = day_totals[key]
        pct_diff = (actual - target) / target * 100
        if pct_diff <= -SUPPLEMENT_TRIGGER_PCT:
            shortfalls[label] = round(-pct_diff, 1)
        elif pct_diff >= SUPPLEMENT_TRIGGER_PCT:
            overshoots[label] = round(pct_diff, 1)

    if not shortfalls and not overshoots:
        return False, None, {}

    parts = []
    for k, v in shortfalls.items():
        parts.append(f"{v}% short on {k} ({SUPPLEMENT_OPTIONS[k]} would help)")
    for k, v in overshoots.items():
        explanation = OVERSHOOT_EXPLANATIONS.get(k, "the dish pool couldn't get closer at this target")
        parts.append(f"{v}% over on {k} ({explanation})")

    message = "Today's plan is " + "; ".join(parts) + "."
    return True, message, {"short": shortfalls, "over": overshoots}


# ---------------------------------------------------------------------------
# LAYER 4: build one full day's plan (all meals combined)
# ---------------------------------------------------------------------------

def generate_day_plan(dishes_df, daily_targets, diet_type, day_number, recently_used, max_attempts=4):
    meal_targets = split_daily_to_meals(**daily_targets)
    day_plan = {}
    day_totals = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fats_g": 0.0}

    for meal_type_key, target in meal_targets.items():
        meal_type = "lunch_dinner" if meal_type_key in ("lunch", "dinner") else meal_type_key

        best_dishes, best_totals, best_gap = None, None, float("inf")
        for _ in range(max_attempts):
            combo = select_dish_combo(dishes_df, meal_type, diet_type, target, exclude_names=recently_used)
            final_dishes, totals = optimize_portions(combo, target)
            gap = sum(abs(totals[k] - target[k])
                      for k in ["calories", "protein_g", "carbs_g", "fats_g"])
            if gap < best_gap:
                best_dishes, best_totals, best_gap = final_dishes, totals, gap
            if _within_tolerance(totals, target):
                break

        # Per-meal output: just what dishes, how much of each (as a multiple
        # of its base serving). Meal-level calorie/macro totals are still
        # useful mid-pipeline (day_totals rollup below), but per the app's
        # requirements the response only needs name + portion + calories per
        # dish — no meal-level protein/carbs/fats and no day-level totals.
        items = [
            {
                "dish_name": d["dish_name"],
                "portion": d["quantity"],       # multiple of the dish's base serving
                "serving_size": d["base_portion"],  # e.g. "1 bowl (150g)" — what 1x portion means
                "calories": d["calories"],
            }
            for d in best_dishes
        ]
        day_plan[meal_type_key] = {
            "items": items,
        }
        for k in day_totals:
            day_totals[k] += best_totals[k]

        for d in best_dishes:
            recently_used.add(d["dish_name"])

    # Still computed (kept for internal telemetry / future use), but
    # intentionally NOT included in the returned day_plan_summary below.
    _check_supplement_need(day_totals, daily_targets)

    day_plan_summary = {
        "day": day_number,
        "breakfast": day_plan["breakfast"],
        "lunch": day_plan["lunch"],
        "snack": day_plan["snack"],
        "dinner": day_plan["dinner"],
    }
    return day_plan_summary


# ---------------------------------------------------------------------------
# LAYER 5: 7-day plan with shuffle (no-repeat window)
# ---------------------------------------------------------------------------
# NOTE: an earlier version of this file had a pre-flight
# _assess_target_feasibility() heuristic here that guessed whether a target
# would be hard to match, based on its protein/fat ratio alone. It's removed
# — measured 300-sample testing showed it was a poor predictor (it flagged
# ~60% of targets as risky when only ~5% actually deviated significantly).
# _check_supplement_need() above replaced it, but as of this revision neither
# the per-day nor per-week supplement message is surfaced in the API response.

def generate_week_plan(dishes_df, daily_targets, diet_type, no_repeat_window=2, seed=None):
    """
    seed: optional int. Pass a fixed value in automated tests to make dish
    selection deterministic (the day-to-day variety logic uses random.choice
    internally, so the same target can otherwise produce different plans on
    different runs). Leave as None in production — that's what gives users
    day-to-day variety instead of the same meals every week.
    """
    if seed is not None:
        random.seed(seed)

    week_plan = []
    recently_used_by_day = []  # list of sets, one per day

    for day_num in range(1, 8):
        # exclude dishes used in the last `no_repeat_window` days only (not the whole week)
        exclude = set()
        for prev in recently_used_by_day[-no_repeat_window:]:
            exclude |= prev

        day_result = generate_day_plan(dishes_df, daily_targets, diet_type, day_num, exclude)
        week_plan.append(day_result)
        used_today = {
            item["dish_name"]
            for meal_key in ("breakfast", "lunch", "snack", "dinner")
            for item in day_result[meal_key]["items"]
        }
        recently_used_by_day.append(exclude.union(used_today))

    return {
        "diet_type": diet_type,
        "daily_target": daily_targets,
        "week_plan": week_plan,
    }


# ---------------------------------------------------------------------------
# FLASK API
# ---------------------------------------------------------------------------

app = Flask(__name__)
# By default Flask alphabetically sorts JSON response keys, which scrambles
# the natural order (day, breakfast, lunch, snack, dinner) into alphabetical
# order (breakfast, day, dinner...). Turn that off so the response preserves
# the order we build it in.
app.config["JSON_SORT_KEYS"] = False  # Flask < 2.3
try:
    app.json.sort_keys = False  # Flask >= 2.3 (new JSON provider API)
except AttributeError:
    pass
dishes_df = pd.read_csv(DISH_CSV_PATH)


@app.route("/generate-diet-plan", methods=["POST"])
def generate_diet_plan_endpoint():
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        required = ["total_calories", "protein_g", "carbs_g", "fats_g", "diet_type"]
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        diet_type = data["diet_type"]
        if diet_type not in ("veg", "non-veg"):
            return jsonify({"error": "diet_type must be 'veg' or 'non-veg'"}), 400

        numeric_fields = ["total_calories", "protein_g", "carbs_g", "fats_g"]
        for field in numeric_fields:
            val = data[field]
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                return jsonify({"error": f"{field} must be a number"}), 400
            if val <= 0:
                return jsonify({"error": f"{field} must be greater than 0"}), 400

        # sanity bounds — reject clearly unrealistic values rather than silently
        # producing a broken/empty plan
        if not (800 <= data["total_calories"] <= 6000):
            return jsonify({"error": "total_calories out of realistic range (800-6000)"}), 400
        if not (20 <= data["protein_g"] <= 400):
            return jsonify({"error": "protein_g out of realistic range (20-400)"}), 400

        daily_targets = {
            "total_calories": data["total_calories"],
            "protein_g": data["protein_g"],
            "carbs_g": data["carbs_g"],
            "fats_g": data["fats_g"],
        }

        # Optional: pass {"seed": 42} in the request body for deterministic
        # output (useful for the app dev's own QA/regression tests). Omit in
        # normal production traffic so users get day-to-day meal variety.
        seed = data.get("seed")
        if seed is not None and not isinstance(seed, int):
            return jsonify({"error": "seed must be an integer"}), 400

        result = generate_week_plan(dishes_df, daily_targets, diet_type, seed=seed)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Basic liveness check for load balancers / uptime monitors."""
    return jsonify({"status": "ok", "dishes_loaded": len(dishes_df)})


if __name__ == "__main__":
    # Dev-only entrypoint. In production, run behind a real WSGI server, e.g.:
    #   gunicorn -w 4 -b 0.0.0.0:5001 diet_plan_api:app
    # debug=True must NEVER be on in production — it enables the Werkzeug
    # interactive debugger, which allows arbitrary code execution if reached.
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=debug_mode, host="0.0.0.0", port=port)