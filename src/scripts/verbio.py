# Functions for preprocessing the VerBIO dataset

# References:
# https://hubbs.engr.tamu.edu/resources/verbio-dataset/

from argparse import ArgumentParser
import os
from pathlib import Path
import pandas as pd
import re
import logging
from utils.scales import AnxietyLevel

target_dirs = ["POST", "PRE"]


def get_anxiety_level(score: int) -> AnxietyLevel:
    """
    Returns the anxiety level of a given full STAI score based on these thresholds:

    - [20, 27] = 0 - no or low anxiety
    - [38, 44] = 1 - moderate anxiety
    - [45, 80] = 2 - high anxiety

    :param score: Full STAI score.
    :return: Anxiety level.
    """
    if score <= 27:
        return AnxietyLevel.NoOrLow
    if score <= 44:
        return AnxietyLevel.Moderate
    return AnxietyLevel.High


def label_anxiety(verbio_path: str) -> None:
    """
    Labels the STAI questionnaire scores in the VerBIO dataset with levels of anxiety.

    :param verbio_path: Path to the VerBIO dataset.
    """
    if not os.path.exists(verbio_path):
        logging.error(f"Path {verbio_path} does not exist.")
        return

    for target_dir in target_dirs:
        target_path = os.path.join(verbio_path, target_dir)

        if not os.path.exists(target_path):
            logging.warning(f"Path {target_path} does not exist.")
            continue

        report_path = os.path.join(target_path, "Self_Reports")

        if not os.path.exists(report_path):
            logging.warning(
                f"Path {target_path} does not have a Self_Reports directory."
            )
            continue

        for file in next(os.walk(report_path))[2]:
            if re.fullmatch(r"^(POST|PRE)_afterPPT.xlsx$", file) is None:
                continue
            questionnaires = pd.read_excel(os.path.join(report_path, file))

            formatted = questionnaires[["PID", "STAI State Score"]].rename(
                columns={"PID": "pid", "STAI State Score": "stai_score"}
            )
            formatted["anxiety_level"] = formatted["stai_score"].apply(
                get_anxiety_level
            )

            output_path = os.path.join(verbio_path, "STAI_data")
            Path(output_path).mkdir(parents=True, exist_ok=True)

            formatted.to_csv(
                os.path.join(output_path, f"{target_dir}_formatted.csv"), index=False
            )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Script for labeling anxiety using the STAI questionnaire scores in the VerBIO dataset."
    )
    parser.add_argument("path", type=str, help="Path to the VerBIO dataset")
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    args = parser.parse_args()
    label_anxiety(args.path)
