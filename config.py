NUM_CLIENTS = 10
NUM_CLASSES = 10
INPUT_DIM = 3072
HIDDEN_DIM = 100

WEIGHT_SCALE = 0.1
RHO_OFFSET = -3.0
ZETA = 10.0

LOCAL_EPOCHS = 5
BATCH_SIZE = 100
GLOBAL_LR = 0.001
PERSONAL_LR = 0.001
MC_SAMPLES = 3

CORESET_SIZE = 200
CORESET_DRAWS = 50
IHT_ITERS = 300
IHT_TOL = 1e-5

WINDOW = 1000
ARRIVALS = 100
STREAM_START = 20

FIRST_DRIFT = dict(out_start=30, out_end=49, back_start=65, back_end=84, target=[0.1, 0.3, 0.6])
SECOND_DRIFT = dict(out_start=105, out_end=124, back_start=140, back_end=159, target=[0.6, 0.3, 0.1])

SCHEDULES = {
    "one_cycle": dict(
        rounds=105,
        cycles=[FIRST_DRIFT],
        windows=[("last 10 rounds", 95, 105)],
    ),
    "two_cycle": dict(
        rounds=180,
        cycles=[FIRST_DRIFT, SECOND_DRIFT],
        windows=[
            ("first drift, drifted phase", 50, 65),
            ("after first drift", 95, 105),
            ("second drift, drifted phase", 125, 140),
            ("after second drift", 170, 180),
        ],
    ),
}

STAT_SIZE = 500
UP_SIZE = 200
DOWN_SIZE = 200
PHIST_SIZE = 150
GLOBAL_PER_CLASS = 100

ARMS = {
    "FULL": dict(method="pfedbayes", solve="never", cap=None, uniform=False, protect=False),
    "B": dict(method="randomsubset", solve="never", cap=None, uniform=True, protect=False),
    "BAL": dict(method="balanced", solve="never", cap=None, uniform=True, protect=False),
    "RES": dict(method="reservoir", solve="never", cap=None, uniform=True, protect=True),
    "C": dict(method="coreset", solve="always", cap=None, uniform=False, protect=False),
    "C80": dict(method="coreset", solve="always", cap=80, uniform=False, protect=False),
    "C40": dict(method="coreset", solve="always", cap=40, uniform=False, protect=False),
    "C20": dict(method="coreset", solve="always", cap=20, uniform=False, protect=False),
    "C0": dict(method="coreset", solve="always", cap=0, uniform=False, protect=False),
    "C0U": dict(method="coreset", solve="always", cap=0, uniform=True, protect=False),
    "P23": dict(method="coreset", solve="skip_third", cap=None, uniform=False, protect=False),
    "D1A": dict(method="coreset", solve="once", cap=None, uniform=False, protect=False),
    "D1B": dict(method="coreset", solve="once", cap=None, uniform=False, protect=True),
    "D1C": dict(method="coreset", solve="always", cap=None, uniform=True, protect=False),
}
