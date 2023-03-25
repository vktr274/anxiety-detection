# Functions for preprocessing the WESAD dataset

import os
import pandas as pd

type = {
    "positive": {
        # level : weight
        1: 4,
        2: 3,
        3: 2,
        4: 1,
    },
    "negative": {
        # level : weight
        1: 1,
        2: 2,
        3: 3,
        4: 4,
    },
}

# 0 = 'I feel at ease'
# 1 = 'I feel nervous'
# 2 = 'I am jittery'
# 3 = 'I am relaxed'
# 4 = 'I am worried'
# 5 = 'I feel pleasant'
weights = {
    # item : type
    0: type["positive"],
    1: type["negative"],
    2: type["negative"],
    3: type["positive"],
    4: type["negative"],
    5: type["positive"],
}


def getWeight(item: int, level: int) -> int:
    """
    Returns the weight of a given item and level.
    :param item: Item number.
    :param level: Level number.
    :return: Weight of the item and level.
    """
    return weights[item][level]


def getScore(row: "pd.Series[int]") -> int:
    """
    Returns the score of a given list.
    :param row: Row of the questionnaire.
    :return: Score of the row.
    """
    score = 0
    for item in range(6):
        score += getWeight(item, row.iloc[item])
    return score


def formatWesad(path: str) -> None:
    """
    Function for labeling anxiety using the STAI questionnaire responses in the WESAD dataset.
    :param path: Path to the WESAD dataset.
    :return: None
    """
    subjects = next(os.walk(path))[1]
    for subject in subjects:
        print(f"Processing subject {subject}")
        questionnaire = pd.read_csv(
            os.path.join(path, subject, f"{subject}_quest.csv"), sep=";", header=None
        )
        questionnaire[0].replace("# ", "", inplace=True, regex=True)
        condition = questionnaire.iloc[1].dropna()
        condition = condition[~condition.str.contains("ORDER|bRead|sRead|fRead")]

        stai = questionnaire[questionnaire[0] == "STAI"].dropna(axis=1)
        stai = stai.iloc[:, 1:].astype(int).apply(getScore, axis=1)

        start = questionnaire[questionnaire[0] == "START"].dropna(axis=1)
        start = start.iloc[:, 1:].astype(float).iloc[:, : len(condition)]

        end = questionnaire[questionnaire[0] == "END"].dropna(axis=1)
        end = end.iloc[:, 1:].astype(float).iloc[:, : len(condition)]

        formatted = pd.DataFrame(
            {
                "condition": condition.values.ravel(),
                "start": start.values.ravel(),
                "end": end.values.ravel(),
                "stai_score": stai.values.ravel(),
            }
        )
        formatted["anxiety_level"] = formatted["stai_score"].apply(
            lambda x: "0" if x <= 11 else "1" if x <= 13 else "2"
        )
        formatted.to_csv(
            os.path.join(path, subject, f"{subject}_STAI.csv"), index=False
        )


if __name__ == "__main__":
    formatWesad("data/WESAD")
