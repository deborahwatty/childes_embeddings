import os
import re
from collections import Counter
import pandas as pd
from collections import defaultdict

children = ['CHI', 'CHI3', 'CHI2', 'CHI1', 'CH2']
adults = ['MOT', 'EXP', 'FAT', 'GFA' 'GMO', 'AUT', 'GMO1', 'GMO2', 'ADU', 'BAO', 'AYY', 'OFF', 'OMM', 'OYY', 'NNN', 'OBB', 'CAA', 'OPP', 'MNN', 'ONN', 'LYY', 'LLL', 'TEA', 'REC', 'CAM', 'GRA', 'SHE', 'GRF', 'GRM', 'EXP1', 'EXP2', 'EXA', 'GFT', 'GMT', 'DAD', 'CAR', 'NAN', 'AEX', 'TSA', 'FRE', 'LIA', 'ANO', 'INV', 'GFA', 'GMO', 'EX1', 'EX2', 'GF']
other = ['SIS', 'NEI', 'YYY', 'UNC', 'CH2', 'JJJ', 'SSS', 'DDD', 'DGG', 'JEE', 'OGG', 'XGG', 'RED', 'STU', 'SI1', 'SI2', 'SI3', 'VIS', 'SHO', 'AUN', 'BRO', 'UNI']


def create_participant_dict(cha_path):
    """
    :param cha_path: Path to a .cha file from the CHILDES corpus
    :return: Dictionary with three keys (adu, chi, unknown). Each list has tuples of participant codes and their ages.
    """


    chi = []
    adu = []
    unknown = []

    def parse_age(age_str):
        try:
            y, _ = age_str.strip(".").split(";")
            return int(y)
        except:
            return None

    with open(cha_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    id_lines = [line.strip() for line in lines if line.startswith("@ID:")]

    for line in id_lines:
        parts = line.split("|")
        if len(parts) < 5:
            continue  # skip malformed lines

        code = parts[2].strip()
        age_raw = parts[3].strip()
        age = parse_age(age_raw)
        role = line.strip()

        participant_info = (code, age)

        if code in children:
            chi.append(participant_info)
        elif code in adults:
            adu.append(participant_info)
        elif code in other and age is not None:
            if age < 16:
                chi.append(participant_info)
            else:
                adu.append(participant_info)
        elif code in other and age is None:
            if code == "BRO":
                if "Child" in role:
                    chi.append(participant_info)
                elif "female" in role:
                    unknown.append(participant_info)
                else:
                    unknown.append(participant_info)
            elif code == "NEI":
                if "Investigator" in role:
                    adu.append(participant_info)
                else:
                    unknown.append(participant_info)
            elif code == "AUN":
                if "Adult" in role:
                    adu.append(participant_info)
                else:
                    unknown.append(participant_info)
            elif code == "YYY":
                if "Grandfather" in role:
                    adu.append(participant_info)
                elif "Child" in role:
                    chi.append(participant_info)
                else:
                    unknown.append(participant_info)
            elif code == "MOT":
                if "male" in role or (age is not None and age < 18):
                    unknown.append(participant_info)
                else:
                    adu.append(participant_info)
            elif code == "CH2":
                if "Mother" in role:
                    unknown.append(participant_info)
                else:
                    chi.append(participant_info)
            elif code == "JJJ":
                if "Unidentified" in role:
                    unknown.append(participant_info)
                else:
                    adu.append(participant_info)
            elif code == "DGG":
                if "Mother" in role:
                    adu.append(participant_info)
                else:
                    unknown.append(participant_info)
            elif code == "SHO":
                if "Target_Child" in role:
                    chi.append(participant_info)
                elif "Investigator" in role:
                    adu.append(participant_info)
                else:
                    unknown.append(participant_info)
            elif code in [
                "UNC", "SSS", "DDD", "JEE", "OGG", "XGG", "RED",
                "STU", "SI1", "SI2", "SI3", "VIS", "UNI"
            ]:
                unknown.append(participant_info)
            else:
                unknown.append(participant_info)
        else:
            print(f"Unrecognized participant code: {code} \n {role}")
            unknown.append(participant_info)

    return {"chi": chi, "adu": adu, "unknown": unknown}


def extract_child_utterances_by_age(cha_path):
    """
    Extracts child utterances grouped by age (in years) from a .cha file.

    :param cha_path: Path to .cha file
    :return: dict[age_in_years] = list of utterances (str)
    """
    participant_info = create_participant_dict(cha_path)
    code_to_age = {code: age for code, age in participant_info["chi"] if age is not None}

    utterances_by_age = defaultdict(list)

    with open(cha_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("*"):
                speaker = line[1:4]
                if speaker in code_to_age:
                    age = code_to_age[speaker]
                    utterance = line.split(":", 1)[1].strip()
                    utterances_by_age[age].append(utterance)

    return dict(utterances_by_age)


def extract_adult_utterances(cha_path):
    """
    Extracts utterances from adult speakers in a CHAT file.

    :param cha_path: Path to .cha file
    :return: list of utterances (str)
    """
    participant_info = create_participant_dict(cha_path)
    adult_codes = {code for code, _ in participant_info["adu"]}

    utterances = []

    with open(cha_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("*"):
                speaker = line[1:4]
                if speaker in adult_codes:
                    utterance = line.split(":", 1)[1].strip()
                    utterances.append(utterance)

    return utterances



def extract_utterances_by_age_and_adult(cha_path):
    """
    Combines child utterances (by age) and all adult utterances.

    :param cha_path: Path to .cha file
    :return: dict with keys:
             - "child": dict {age (int): list of utterances}
             - "adult": list of utterances
    """
    child_utterances = extract_child_utterances_by_age(cha_path)
    adult_utterances = extract_adult_utterances(cha_path)

    return {
        "child": child_utterances,
        "adult": adult_utterances
    }

import re

def clean_chinese_utterances_simple(data):
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')

    def clean(text):
        return ' '.join(chinese_char_pattern.findall(text))

    cleaned_data = {
        "child": {},
        "adult": []
    }

    # Clean child utterances
    for age, utts in data["child"].items():
        cleaned_data["child"][age] = [
            clean(utt) for utt in utts if clean(utt)
        ]

    # Clean adult utterances
    cleaned_data["adult"] = [
        clean(utt) for utt in data["adult"] if clean(utt)
    ]

    return cleaned_data



def get_child_utterance_gra_by_age(cha_path):
    """
    Extracts child utterances and their %gra dependencies from a .cha file.
    Groups them by age in years. Returns a dictionary:
        {age: [(utterance, gra), ...]}
    Ignores children without age.
    """
    result = defaultdict(list)
    participant_info = create_participant_dict(cha_path)
    chi_codes_with_age = {code: age for code, age in participant_info["chi"] if age is not None}

    with open(cha_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_speaker = None
    current_utt = None

    for i, line in enumerate(lines):
        if line.startswith("*"):
            speaker_code = line[1:4]
            if speaker_code in chi_codes_with_age:
                current_speaker = speaker_code
                current_utt = line[5:].strip()
        elif line.startswith("%gra:") and current_speaker:
            gra = line[5:].strip()
            age = chi_codes_with_age[current_speaker]
            result[age].append((current_utt, gra))
            current_speaker = None  # Reset until next utterance
            current_utt = None

    return result

def get_adult_utterance_gra(cha_path):
    """
    Extracts adult utterances and their %gra dependencies from a .cha file.
    Returns a list of (utterance, gra) tuples.
    """
    result = []
    participant_info = create_participant_dict(cha_path)
    adu_codes = {code for code, _ in participant_info["adu"]}

    with open(cha_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_speaker = None
    current_utt = None

    for line in lines:
        if line.startswith("*"):
            speaker_code = line[1:4]
            if speaker_code in adu_codes:
                current_speaker = speaker_code
                current_utt = line[5:].strip()
            else:
                current_speaker = None
        elif line.startswith("%gra:") and current_speaker:
            gra = line[5:].strip()
            result.append((current_utt, gra))
            current_speaker = None
            current_utt = None

    return result


def get_all_utterance_gra_by_group(cha_path):
    """
    Returns a dictionary:
    {
        "child": { age: [(utterance, gra), ...], ... },
        "adult": [(utterance, gra), ...]
    }
    """
    return {
        "child": get_child_utterance_gra_by_age(cha_path),
        "adult": get_adult_utterance_gra(cha_path)
    }




def extract_utterances_by_age_and_adult_folder(folder_path):
    """
    Walks folder and subfolders, extracts child/adult utterances from all .cha files.
    Returns:
      {
        "child": { age: [utterances, ...], ... },
        "adult": [utterances, ...]
      }
    """
    aggregated = {
        "child": defaultdict(list),
        "adult": []
    }

    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.endswith(".cha"):
                filepath = os.path.join(root, fname)
                data = extract_utterances_by_age_and_adult(filepath)

                # Merge child dicts by age
                for age, utts in data["child"].items():
                    aggregated["child"][age].extend(utts)
                aggregated["adult"].extend(data["adult"])

    return dict(aggregated)


def get_all_utterance_gra_by_group_folder(folder_path):
    """
    Walks folder and subfolders, extracts child/adult utterance+gra tuples from all .cha files.
    Returns:
      {
        "child": { age: [(utterance, gra), ...], ... },
        "adult": [(utterance, gra), ...]
      }
    """
    aggregated = {
        "child": defaultdict(list),
        "adult": []
    }

    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.endswith(".cha"):
                filepath = os.path.join(root, fname)
                data = get_all_utterance_gra_by_group(filepath)

                for age, tuples in data["child"].items():
                    aggregated["child"][age].extend(tuples)
                aggregated["adult"].extend(data["adult"])

    return dict(aggregated)



def clean_chinese_utterances(data):
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')

    def clean(text):
        return ' '.join(chinese_char_pattern.findall(text))

    cleaned_data = {
        "child": {},
        "adult": []
    }

    # Clean child utterances
    for age, utts in data["child"].items():
        cleaned_data["child"][age] = [
            (clean(utt), gra) for utt, gra in utts if clean(utt)
        ]

    # Clean adult utterances
    cleaned_data["adult"] = [
        (clean(utt), gra) for utt, gra in data["adult"] if clean(utt)
    ]

    return cleaned_data


