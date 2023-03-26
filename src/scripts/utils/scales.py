from enum import IntEnum

class AnxietyLevel(IntEnum):
    NoneOrLow = 0
    Moderate = 1
    High = 2


class Feeling(IntEnum):
    AtEase = 0
    Nervous = 1
    Jittery = 2
    Relaxed = 3
    Worried = 4
    Pleasant = 5


class LikertScale4(IntEnum):
    NotAtAll = 1
    Somewhat = 2
    Moderately = 3
    Very = 4