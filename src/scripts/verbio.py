# Functions for processing the VerBIO dataset

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


def process_verbio(verbio_path: str) -> None:
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

        logging.info(f"Processing {target_dir}...")

        report_path = os.path.join(target_path, "Self_Reports")
        actiwave_path = os.path.join(target_path, "Actiwave")

        if not os.path.exists(report_path):
            logging.warning(
                f"Path {target_path} does not have a Self_Reports directory."
            )
            continue

        files = next(os.walk(report_path))[2]
        matches = list(
            filter(lambda x: re.fullmatch(r"^(POST|PRE)_afterPPT.xlsx$", x), files)
        )

        if len(matches) == 0:
            logging.warning(
                f"Path {target_path} does not have a Self_Reports directory with a questionnaire file."
            )
            continue
        elif len(matches) > 1:
            logging.warning(
                f"Path {target_path} has multiple questionnaire files. Using the first one."
            )

        questionnaire_file = matches[0]
        questionnaires = pd.read_excel(os.path.join(report_path, questionnaire_file))

        formatted = questionnaires[["PID", "STAI State Score"]].rename(
            columns={"PID": "pid", "STAI State Score": "stai_score"}
        )
        formatted["anxiety_level"] = formatted["stai_score"].apply(get_anxiety_level)

        output_path = os.path.join(verbio_path, "Processed")
        Path(output_path).mkdir(parents=True, exist_ok=True)

        questionnaires_output_path = os.path.join(output_path, f"{target_dir}_q.csv")
        logging.info(
            f"Writing processed questionnaires to {questionnaires_output_path}"
        )
        formatted.to_csv(questionnaires_output_path, index=False)

        for pid in next(os.walk(actiwave_path))[1]:
            pid_path = os.path.join(actiwave_path, pid)
            ecg_ppt_path = os.path.join(pid_path, "ECG_PPT.xlsx")
            if not os.path.exists(ecg_ppt_path):
                logging.warning(f"Path {pid_path} does not have an ECG_PPT.xlsx file.")
                continue
            ecg = pd.read_excel(ecg_ppt_path)
            ecg = ecg[["ECG"]]
            pid_row = formatted[formatted["pid"] == pid]
            if pid_row.empty:
                logging.warning(f"Could not find anxiety level for {pid}.")
                continue
            anxiety_level: int = pid_row.iloc[0]["anxiety_level"]
            ecg_output_path = os.path.join(
                output_path, f"{pid}_PPT_{target_dir}_{anxiety_level}.csv"
            )
            logging.info(
                f"Writing ECG for {pid} with anxiety level {anxiety_level} to {ecg_output_path}"
            )
            ecg.to_csv(ecg_output_path, index=False)


def create_cli() -> ArgumentParser:
    """
    Creates a CLI for the script.

    :return: CLI argument parser.
    """
    parser = ArgumentParser(
        description="Script for labeling measurements with anxiety levels using the shortened STAI questionnaire responses in the VerBIO dataset."
    )
    parser.add_argument("path", type=str, help="Path to the VerBIO dataset")
    return parser


def main():
    cli = create_cli()
    args = cli.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    process_verbio(args.path)


if __name__ == "__main__":
    main()
