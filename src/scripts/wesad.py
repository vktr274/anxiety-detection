# Functions for preprocessing the WESAD dataset

# References:
# https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/

import os
import pandas as pd
from argparse import ArgumentParser
import logging
from pathlib import Path
import re
from utils.scales import AnxietyLevel, Feeling, LikertScale4

type = {
    "positive": {
        # level : weight
        LikertScale4.NotAtAll: 4,
        LikertScale4.Somewhat: 3,
        LikertScale4.Moderately: 2,
        LikertScale4.Very: 1,
    },
    "negative": {
        # level : weight
        LikertScale4.NotAtAll: 1,
        LikertScale4.Somewhat: 2,
        LikertScale4.Moderately: 3,
        LikertScale4.Very: 4,
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
    return weights[Feeling(item)][LikertScale4(level)]


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
    Returns the anxiety level of a given shortened STAI score based on these thresholds:

    - [6, 11] = 0 - none or low anxiety
    - [12, 13] = 1 - moderate anxiety
    - [14, 24] = 2 - high anxiety

    :param score: Shortened STAI score.
    :return: Anxiety level.
    """
    if score <= 11:
        return AnxietyLevel.NoOrLow
    if score <= 13:
        return AnxietyLevel.Moderate
    return AnxietyLevel.High


def label_anxiety(wesad_path: str) -> None:
    """
    Function for labeling anxiety using the shortened
    STAI questionnaire responses in the WESAD dataset.

    The function will create a new CSV file for each subject
    with the following columns:
        - condition: condition in the experiment
        - start: start time of the condition
        - end: end time of the condition
        - stai_score: shortened STAI score
        - anxiety_level: anxiety level based on the shortened STAI score

    :param path: Path to the WESAD dataset.
    :return: None
    """
    if not os.path.exists(wesad_path):
        logging.error(f"Path '{wesad_path}' does not exist.")
        return

    subjects = next(os.walk(wesad_path))[1]
    if len(subjects) == 0:
        logging.error(f"No subjects found in '{wesad_path}'")
        return

    for subject in subjects:
        if subject == "STAI_data":
            continue
        if re.fullmatch(r"^S[0-9]{1,2}$", subject) is None:
            logging.warning(f"Subject {subject} does not have a valid name.\n")
            continue
        logging.info(f"Processing subject {subject}")

        questionnaire_path = os.path.join(wesad_path, subject, f"{subject}_quest.csv")
        if not os.path.exists(questionnaire_path):
            logging.warning(f"Subject {subject} does not have a questionnaire file.\n")
            continue

        questionnaire = pd.read_csv(
            questionnaire_path,
            sep=";",
            header=None,
        )
        questionnaire[0].replace("# ", "", inplace=True, regex=True)

        condition = questionnaire.iloc[1].dropna().astype(str)
        condition = condition[
            ~condition.str.contains("ORDER|bRead|sRead|fRead")
        ].astype(str)

        stai = questionnaire[questionnaire[0] == "STAI"].dropna(axis=1)
        stai_score = stai.iloc[:, 1:].astype(int).apply(get_score, axis=1)

        start = questionnaire[questionnaire[0] == "START"].dropna(axis=1)
        start = start.iloc[:, 1 : len(condition) + 1].astype(float)

        end = questionnaire[questionnaire[0] == "END"].dropna(axis=1)
        end = end.iloc[:, 1 : len(condition) + 1].astype(float)

        formatted = pd.DataFrame(
            {
                "condition": condition.values,
                "start": start.values.ravel(),
                "end": end.values.ravel(),
                "stai_score": stai_score.values,
            }
        )
        formatted["anxiety_level"] = formatted["stai_score"].apply(get_anxiety_level)

        output_path = os.path.join(wesad_path, "STAI_data")
        Path(output_path).mkdir(parents=True, exist_ok=True)

        formatted.to_csv(os.path.join(output_path, f"{subject}_STAI.csv"), index=False)

        logging.info(f"Anxiety levels: {formatted['anxiety_level'].ravel()}")
        logging.info(f"Saved to '{os.path.join(output_path, subject)}_STAI.csv'\n")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Script for labeling anxiety using the shortened STAI questionnaire responses in the WESAD dataset."
    )
    parser.add_argument("path", type=str, help="Path to the WESAD dataset")
    args = parser.parse_args()
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    label_anxiety(args.path)
