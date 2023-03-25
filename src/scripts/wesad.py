# Functions for preprocessing the WESAD dataset

# References:
# https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/

from enum import IntEnum
import os
import pandas as pd


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


class LikertScale(IntEnum):
    NotAtAll = 1
    Somewhat = 2
    Moderately = 3
    Very = 4


type = {
    "positive": {
        # level : weight
        LikertScale.NotAtAll: 4,
        LikertScale.Somewhat: 3,
        LikertScale.Moderately: 2,
        LikertScale.Very: 1,
    },
    "negative": {
        # level : weight
        LikertScale.NotAtAll: 1,
        LikertScale.Somewhat: 2,
        LikertScale.Moderately: 3,
        LikertScale.Very: 4,
    },
}

weights = {
    Feeling.AtEase: type["positive"],
    Feeling.Nervous: type["negative"],
    Feeling.Jittery: type["negative"],
    Feeling.Relaxed: type["positive"],
    Feeling.Worried: type["negative"],
    Feeling.Pleasant: type["positive"],
}


def get_weight(item: int, level: int) -> int:
    """
    Returns the weight of a given item and level.
    
    :param item: Item number.
    :param level: Level number.
    :return: Weight of the item and level.
    """
    return weights[Feeling(item)][LikertScale(level)]


def get_score(stai6: "pd.Series[int]") -> int:
    """
    Returns the score of a given questionnaire.

    :param stai6: Shortened STAI questionnaire.
    :return: Score of the shortened STAI questionnaire.
    """
    score = 0
    for item, level in enumerate(stai6):
        score += get_weight(item, level)
    return score


def get_anxiety_level(score: int) -> AnxietyLevel:
    """
    Returns the anxiety level of a given score based on these thresholds:

    - [6, 11] = 0 - none or low anxiety
    - [12, 13] = 1 - moderate anxiety
    - [14, 24] = 2 - high anxiety

    :param score: Shortened STAI score.
    :return: Anxiety level.
    """
    if score <= 11:
        return AnxietyLevel.NoneOrLow
    if score <= 13:
        return AnxietyLevel.Moderate
    return AnxietyLevel.High


def label_anxiety(wesad_path: str) -> None:
    """
    Function for labeling anxiety using the shortened
    STAI questionnaire responses in the WESAD dataset.

    :param path: Path to the WESAD dataset.
    :return: None
    """
    subjects = next(os.walk(wesad_path))[1]
    for subject in subjects:
        print(f"Processing subject {subject}")
        questionnaire = pd.read_csv(
            os.path.join(wesad_path, subject, f"{subject}_quest.csv"),
            sep=";",
            header=None,
        )
        questionnaire[0].replace("# ", "", inplace=True, regex=True)
        condition = questionnaire.iloc[1].dropna()
        condition = condition[~condition.str.contains("ORDER|bRead|sRead|fRead")]

        stai = questionnaire[questionnaire[0] == "STAI"].dropna(axis=1)
        stai_score = stai.iloc[:, 1:].astype(int).apply(get_score, axis=1)

        start = questionnaire[questionnaire[0] == "START"].dropna(axis=1)
        start = start.iloc[:, 1:].astype(float).iloc[:, : len(condition)]

        end = questionnaire[questionnaire[0] == "END"].dropna(axis=1)
        end = end.iloc[:, 1:].astype(float).iloc[:, : len(condition)]

        formatted = pd.DataFrame(
            {
                "condition": condition.values.ravel(),
                "start": start.values.ravel(),
                "end": end.values.ravel(),
                "stai_score": stai_score.values.ravel(),
            }
        )
        formatted["anxiety_level"] = formatted["stai_score"].apply(get_anxiety_level)
        formatted.to_csv(
            os.path.join(wesad_path, subject, f"{subject}_STAI.csv"), index=False
        )


if __name__ == "__main__":
    label_anxiety("data/WESAD")
