# Functions for processing the WESAD dataset

# References:
# https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/

import os
import sys
import pandas as pd
from argparse import ArgumentParser
import logging
from pathlib import Path
import re
from io import TextIOWrapper
from typing import Any, Callable
import json
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


def get_metadata(header: list[str]) -> tuple[str, list[str]]:
    """
    Returns the metadata of a given header.

    :param header: Header of the respiban file.
    :return: Tuple of the start time and the sensors.
    """
    meta: dict[str, Any] = json.loads(header[1].split("#", maxsplit=1)[1])
    metadata: dict[str, Any] = meta[list(meta.keys())[0]]
    return metadata["time"], metadata["sensor"]


def read_respiban_df(
    file: TextIOWrapper,
    sensors: list[str],
    conversions: dict[str, Callable[[int], float]],
) -> pd.DataFrame:
    """
    Returns a DataFrame of the respiban file with the desired converted columns.

    :param file: File object of the respiban file.
    :param sensors: List of sensors in the respiban file.
    :param conversions: Dictionary of conversions of the raw data to the desired units.
    :return: DataFrame of the respiban file with the desired converted columns.
    """
    df = (
        pd.read_csv(file, sep="\t", header=None, index_col=0)
        .dropna(axis=1)
        .drop(1, axis=1)
    )
    df.columns = sensors
    df.index.name = None
    columns = list(conversions.keys())
    result_df = pd.DataFrame(columns=columns)
    for column in columns:
        result_df[column] = df[column].apply(conversions[column])
    return result_df


def read_subject(
    path: str,
    subject: str,
    conversions: dict[str, Callable[[int], float]],
) -> tuple[str, pd.DataFrame]:
    """
    Returns the start time and DataFrame of the respiban file of a given subject.

    :param path: Path to the WESAD dataset.
    :param subject: Subject ID.
    :param conversions: Dictionary of conversions of the raw data to the desired units.
    :return: Tuple of the start time and the DataFrame of the respiban file.
    """
    file_path = os.path.join(path, subject, f"{subject}_respiban.txt")
    if not os.path.exists(file_path):
        logging.error(f"{subject}: subject does not have a respiban file")
        sys.exit(1)
    with open(os.path.join(path, subject, f"{subject}_respiban.txt")) as f:
        header = [f.readline() for _ in range(3)]
        start_time, sensors = get_metadata(header)
        df = read_respiban_df(f, sensors, conversions)
    return start_time, df


def process_wesad(
    wesad_path: str,
    conversions: dict[str, Callable[[int], float]],
    sampling_rate: int,
) -> None:
    """
    Function for labeling measurements with anxiety levels using the shortened
    STAI questionnaire responses in the WESAD dataset.

    The function will create a 6 new CSV files for each subject, one for the STAI
    questionnaire results and one for each of the 5 conditions in the experiment:

    - <subject>_STAI.csv: DataFrame with the following columns:
        - condition: condition in the experiment
        - start: offset from the start of the experiment where the condition starts
        - end: offset from the start of the experiment where the condition ends
        - stai_score: shortened STAI score
        - anxiety_level: anxiety level based on the shortened STAI score
    - <subject>_<condition>_<anxiety_level>.csv: DataFrame with measurements
    converted to the desired units, each in a separate column

    :param path: Path to the WESAD dataset.
    :param conversions: Dictionary of conversions of the raw data to the desired units.
    :param sampling_rate: Sampling rate of the RespiBAN measurements.
    :return: None
    """
    if not os.path.exists(wesad_path):
        logging.error(f"Path '{wesad_path}' does not exist.")
        sys.exit(1)

    subjects = next(os.walk(wesad_path))[1]
    if len(subjects) == 0:
        logging.error(f"No subjects found in '{wesad_path}'")
        sys.exit(1)

    for subject in subjects:
        # Skip the processed folder and subjects with invalid names
        if subject == "Processed":
            continue
        if re.fullmatch(r"^S[0-9]{1,2}$", subject) is None:
            logging.warning(f"{subject}: subject does not have a valid name.\n")
            continue
        logging.info(f"{subject}: processing subject")

        questionnaire_path = os.path.join(wesad_path, subject, f"{subject}_quest.csv")
        if not os.path.exists(questionnaire_path):
            logging.warning(f"{subject}: subject does not have a questionnaire file.\n")
            continue

        # Read questionnaire and get condition, STAI score,
        # START and END and determine the anxiety level.
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
        start = start.iloc[:, 1 : len(condition) + 1].astype(str)

        end = questionnaire[questionnaire[0] == "END"].dropna(axis=1)
        end = end.iloc[:, 1 : len(condition) + 1].astype(str)

        formatted = pd.DataFrame(
            {
                "condition": condition.values,
                "start": start.values.ravel(),
                "end": end.values.ravel(),
                "stai_score": stai_score.values,
            }
        )
        formatted["anxiety_level"] = formatted["stai_score"].apply(get_anxiety_level)

        output_path = os.path.join(wesad_path, "Processed")
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Save STAI questionnaire results to CSV file.
        stai_outut = os.path.join(output_path, f"{subject}_STAI.csv")
        formatted.to_csv(stai_outut, index=False)

        logging.info(
            f"{subject}: anxiety levels - {formatted['anxiety_level'].ravel()}"
        )
        logging.info(f"{subject}: STAI information saved to '{stai_outut}'")

        # Read respiban file and get the start time and DataFrame of measurements.
        _, respiban = read_subject(wesad_path, subject, conversions)

        # Seperate measurements into conditions with anxiety level labels.
        for _, row in formatted.iterrows():
            current_condition: str = row["condition"]
            current_start: str = row["start"]
            current_end: str = row["end"]
            current_anxiety_level: int = row["anxiety_level"]

            # Calculate interval of samples for the condition.
            # The start and end times have a format of 'minutes.seconds'.
            start_split = current_start.split(".")
            start_seconds = int(start_split[0]) * 60 + (
                int(start_split[1]) if len(start_split) > 1 else 0
            )

            end_split = current_end.split(".")
            end_seconds = int(end_split[0]) * 60 + (
                int(end_split[1]) if len(end_split) > 1 else 0
            )

            # Start and end times are multiplied by the sampling rate to get the
            # interval of samples for the condition.
            interval = (start_seconds * sampling_rate, end_seconds * sampling_rate)

            condition_df = respiban.iloc[interval[0] : interval[1] + 1]
            current_condition = current_condition.replace(" ", "")

            condition_output = os.path.join(
                output_path,
                f"{subject}_{current_condition}_{current_anxiety_level}.csv",
            )

            condition_df.to_csv(
                condition_output,
                index=False,
            )

            logging.info(
                f"{subject}: saved {current_condition} with anxiety level {current_anxiety_level} to '{condition_output}'"
            )


def create_cli() -> ArgumentParser:
    """
    Create the CLI parser for the script.

    :return: The CLI parser.
    """
    parser = ArgumentParser(
        description="Script for labeling measurements with anxiety levels using the shortened STAI questionnaire responses in the WESAD dataset."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to the WESAD dataset",
    )
    return parser


def main() -> None:
    cli = create_cli()
    args = cli.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    sampling_rate = 700  # Hz - sampling rate used for the RespiBAN device

    # Conversion constants from the README file of the WESAD dataset
    # used for converting the raw data to the desired units.
    chan_bit = 2**16
    vcc = 3

    # Conversion function for the raw data to the desired units.
    si_conversion = {
        "ECG": lambda x: (x / chan_bit - 0.5) * vcc,
    }

    process_wesad(args.path, si_conversion, sampling_rate)


if __name__ == "__main__":
    main()
