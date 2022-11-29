import math
import numpy as np
import pandas as pd

list_name = ['NOSE_x',
             'NOSE_y',
             'NOSE_score',
             'LEFT_EYE_x',
             'LEFT_EYE_y',
             'LEFT_EYE_score',
             'RIGHT_EYE_x',
             'RIGHT_EYE_y',
             'RIGHT_EYE_score',
             'LEFT_EAR_x',
             'LEFT_EAR_y',
             'LEFT_EAR_score',
             'RIGHT_EAR_x',
             'RIGHT_EAR_y',
             'RIGHT_EAR_score',
             'LEFT_SHOULDER_x',
             'LEFT_SHOULDER_y',
             'LEFT_SHOULDER_score',
             'RIGHT_SHOULDER_x',
             'RIGHT_SHOULDER_y',
             'RIGHT_SHOULDER_score',
             'LEFT_ELBOW_x',
             'LEFT_ELBOW_y',
             'LEFT_ELBOW_score',
             'RIGHT_ELBOW_x',
             'RIGHT_ELBOW_y',
             'RIGHT_ELBOW_score',
             'LEFT_WRIST_x',
             'LEFT_WRIST_y',
             'LEFT_WRIST_score',
             'RIGHT_WRIST_x',
             'RIGHT_WRIST_y',
             'RIGHT_WRIST_score',
             'LEFT_HIP_x',
             'LEFT_HIP_y',
             'LEFT_HIP_score',
             'RIGHT_HIP_x',
             'RIGHT_HIP_y',
             'RIGHT_HIP_score',
             'LEFT_KNEE_x',
             'LEFT_KNEE_y',
             'LEFT_KNEE_score',
             'RIGHT_KNEE_x',
             'RIGHT_KNEE_y',
             'RIGHT_KNEE_score',
             'LEFT_ANKLE_x',
             'LEFT_ANKLE_y',
             'LEFT_ANKLE_score',
             'RIGHT_ANKLE_x',
             'RIGHT_ANKLE_y',
             'RIGHT_ANKLE_score']


def set_center_in_pelvis(df):
    x_names = df.filter(regex='_x').columns
    y_names = df.filter(regex='_y').columns
    x = 0
    y = 0
    for i in range(len(df)):
        if (df.loc[i, 'LEFT_HIP_score'] > 0.3) and (df.loc[i, 'RIGHT_HIP_score'] > 0.3):
            x = ((float(df.loc[i, 'LEFT_HIP_x']) + float(df.loc[i, 'RIGHT_HIP_x'])) / 2)
            y = ((float(df.loc[i, 'LEFT_HIP_y']) + float(df.loc[i, 'RIGHT_HIP_y'])) / 2)
            break
    for i in range(len(df)):
        for j in x_names:
            df.loc[i, j] = (float(df.loc[i, j]) - x)
        for j in y_names:
            df.loc[i, j] = (float((df.loc[i, j])) - y)

    return df


def find_avg_torso(df):
    n = 0
    x = 0
    for i in range(len(df)):
        if ((df.loc[i, 'LEFT_HIP_score'] > 0.3) and (df.loc[i, 'RIGHT_HIP_score'] > 0.3) and (
                df.loc[i, 'LEFT_SHOULDER_score'] > 0.3) and (df.loc[i, 'RIGHT_SHOULDER_score'] > 0.3)):
            x_shoulder = (float(df.loc[i, 'LEFT_SHOULDER_x']) + float(df.loc[i, 'RIGHT_SHOULDER_x'])) / 2
            y_shoulder = (float(df.loc[i, 'LEFT_SHOULDER_y']) + float(df.loc[i, 'RIGHT_SHOULDER_y'])) / 2
            x_hip = (float(df.loc[i, 'LEFT_HIP_x']) + float(df.loc[i, 'RIGHT_HIP_x'])) / 2
            y_hip = (float(df.loc[i, 'LEFT_HIP_y']) + float(df.loc[i, 'RIGHT_HIP_y'])) / 2
            p1 = [x_shoulder, y_shoulder]
            p2 = [x_hip, y_hip]
            m = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
            x = x + 1
            n = n + m
    if x == 0:
        x = 1
    return n / x


def normalize_df(df):
    x_names = df.filter(regex='_x').columns
    y_names = df.filter(regex='_y').columns
    n = find_avg_torso(df)
    for i in range(len(df)):
        for j in x_names:
            df.loc[i, j] = float(df.loc[i, j]) / n
        for k in y_names:
            df.loc[i, k] = float(df.loc[i, k]) / n

    return df


def set_center_and_normalize(path):
    print(f"==================== {path}")
    df = pd.read_csv(path, names=list_name, header=None, delim_whitespace=True)
    print(f"done reading {path} {df.shape}")
    df = df.set_index(df.columns[0]).reset_index()
    # df = delete_low_quality_frames(df)
    df = set_center_in_pelvis(df)
    print(f"done set_center_in_pelvis {path} {df.shape}")
    df = normalize_df(df)
    print(f"done normalizing {path} {df.shape}")
    return df
