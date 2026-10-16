"""Unit tests for gestures.py (waving, pointing, open-hand detection).

MediaPipe gives each hand as 21 landmarks with normalized x/y coordinates.
These tests build fake hands from plain objects, so no webcam, OpenCV, or
MediaPipe is needed. Run from the repo root with:  pytest -v
"""
from types import SimpleNamespace

import pytest

import gestures
from gestures import detectionFrames, isOpenHand, isPointing, isWaving

FINGERS = ["index", "middle", "ring", "pinky"]
# MediaPipe landmark indices: (knuckle-side joint, fingertip) for each finger
FINGER_LANDMARKS = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}


# ---------- helpers: fake hands ----------

def make_hand(extended=FINGERS):
    """A hand with the wrist at the bottom and each finger either extended
    (tip far from the wrist) or curled (tip closer to the wrist than its knuckle)."""
    lm = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
    lm[0] = SimpleNamespace(x=0.5, y=0.9)  # wrist
    for i, finger in enumerate(FINGERS):
        knuckle_idx, tip_idx = FINGER_LANDMARKS[finger]
        x = 0.35 + 0.1 * i
        lm[knuckle_idx] = SimpleNamespace(x=x, y=0.7)
        lm[tip_idx] = SimpleNamespace(x=x, y=0.4 if finger in extended else 0.8)
    return SimpleNamespace(landmark=lm)


def make_wave_hand(y):
    """A hand whose wrist-area landmarks sit at height y (used for wave tests)."""
    lm = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
    lm[0].y = y
    lm[1].y = y
    return SimpleNamespace(landmark=lm)


def feed(ys, label="Right"):
    """Feed a sequence of heights to isWaving; return the last result."""
    result = None
    for y in ys:
        result = isWaving(make_wave_hand(y), label)
    return result


def oscillation(n=detectionFrames, low=0.4, high=0.5):
    """Up-and-down motion; each step (~0.1) is bigger than the movement threshold."""
    return [low if i % 2 == 0 else high for i in range(n)]


@pytest.fixture(autouse=True)
def clear_histories():
    """isWaving keeps module-level history, so reset it around every test."""
    gestures.leftMovementHistory.clear()
    gestures.rightMovementHistory.clear()
    yield
    gestures.leftMovementHistory.clear()
    gestures.rightMovementHistory.clear()


# ---------- isPointing ----------

class TestIsPointing:
    @pytest.mark.parametrize("finger", FINGERS)
    def test_extended_finger_is_pointing(self, finger):
        assert isPointing(make_hand(extended=[finger]), finger) is True

    @pytest.mark.parametrize("finger", FINGERS)
    def test_curled_finger_is_not_pointing(self, finger):
        assert isPointing(make_hand(extended=[]), finger) is False

    @pytest.mark.parametrize("finger", FINGERS)
    def test_only_the_extended_finger_is_detected(self, finger):
        hand = make_hand(extended=[finger])
        results = {f: isPointing(hand, f) for f in FINGERS}
        assert results == {f: (f == finger) for f in FINGERS}

    def test_tip_exactly_as_far_as_knuckle_is_not_pointing(self):
        # Boundary: the comparison is strict (>), so equal distance means "not pointing".
        hand = make_hand(extended=[])
        hand.landmark[8] = SimpleNamespace(x=hand.landmark[6].x, y=hand.landmark[6].y)
        assert isPointing(hand, "index") is False

    def test_unknown_finger_name_raises(self):
        # Characterization test: documents current behaviour. An unrecognised name
        # leaves 'fingertip' unassigned, so Python raises UnboundLocalError.
        with pytest.raises(UnboundLocalError):
            isPointing(make_hand(), "thumb")


# ---------- isOpenHand ----------

class TestIsOpenHand:
    def test_all_fingers_extended_is_open(self):
        assert isOpenHand(make_hand(extended=FINGERS)) is True

    def test_fist_is_not_open(self):
        assert isOpenHand(make_hand(extended=[])) is False

    @pytest.mark.parametrize("curled", FINGERS)
    def test_one_curled_finger_means_not_open(self, curled):
        extended = [f for f in FINGERS if f != curled]
        assert isOpenHand(make_hand(extended=extended)) is False

    def test_only_index_extended_is_not_open(self):
        assert isOpenHand(make_hand(extended=["index"])) is False


# ---------- isWaving ----------

class TestIsWaving:
    def test_not_enough_history_returns_false(self):
        # Even clear waving can't be reported until the window is full.
        for y in oscillation()[: detectionFrames - 1]:
            assert isWaving(make_wave_hand(y), "Right") is False

    def test_full_window_of_big_oscillation_is_a_wave(self):
        assert feed(oscillation()) is True

    def test_still_hand_is_not_a_wave(self):
        assert feed([0.5] * detectionFrames) is False

    def test_small_jitter_below_threshold_is_not_a_wave(self):
        jitter = [0.50 if i % 2 == 0 else 0.52 for i in range(detectionFrames)]
        assert feed(jitter) is False

    def test_one_direction_change_is_not_enough(self):
        # up, up, down, down -> directions [+, +, -, -] -> 1 change
        ys = [0.2] * 21 + [0.3, 0.4, 0.3, 0.2]
        assert len(ys) == detectionFrames
        assert feed(ys) is False

    def test_two_direction_changes_is_just_below_the_limit(self):
        # up, down, up -> 2 changes (limit is 3)
        ys = [0.2] * 22 + [0.3, 0.2, 0.3]
        assert len(ys) == detectionFrames
        assert feed(ys) is False

    def test_three_direction_changes_is_exactly_enough(self):
        # up, down, up, down -> 3 changes
        ys = [0.2] * 21 + [0.3, 0.2, 0.3, 0.2]
        assert len(ys) == detectionFrames
        assert feed(ys) is True

    def test_history_never_grows_past_the_window(self):
        feed([0.5] * (detectionFrames + 10))
        assert len(gestures.rightMovementHistory) == detectionFrames

    def test_left_and_right_hands_are_tracked_separately(self):
        assert feed(oscillation(), label="Left") is True
        assert isWaving(make_wave_hand(0.5), "Right") is False
        assert len(gestures.rightMovementHistory) == 1

    def test_old_waving_scrolls_out_of_the_window(self):
        assert feed(oscillation()) is True
        # A full window of stillness pushes the old motion out entirely.
        assert feed([0.5] * detectionFrames) is False
