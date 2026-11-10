
"""Milestone 1, Q3: construire un CSV reproductible de tirs cadrés et buts."""

import json
from pathlib import Path

import pandas as pd

from ift6758.features.shot_features import extract_shots


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'


def build_dataset():
    files = sorted(RAW.rglob('*.json'))

    if not files:
        raise FileNotFoundError('Aucun JSON trouvé dans data/raw')

    frames = []

    for file in files:
        with file.open(encoding='utf-8') as f:
            frame = extract_shots(json.load(f))

        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise ValueError('Aucun tir cadré ou but trouvé')

    result = pd.concat(frames, ignore_index=True)

    # Vérification des doublons
    duplicates = result.duplicated(
        ['game_id', 'event_id']
    ).sum()

    if duplicates:
        raise ValueError(
            f'{duplicates} événements dupliqués : vérifiez les JSON'
        )

    # Sauvegarde des données
    PROCESSED.mkdir(parents=True, exist_ok=True)

    csv_file = PROCESSED / 'shots.csv'

    result.to_csv(csv_file, index=False)

    print('CSV :', csv_file)

    print('Matchs :', result['game_id'].nunique())

    print(
        'Événements :', len(result),
        '| Buts :', int(result['is_goal'].sum())
    )

    print(
        'Colonnes :', len(result.columns),
        '| Doublons :', duplicates
    )

    print(
        '\nValeurs manquantes :\n',
        result.isna().sum().to_string()
    )

    # Quantifier les exclusions selon les analyses
    print('\nExclusions pour une analyse nécessitant :')

    for needed in (
        ['x_coord', 'y_coord'],
        ['shot_distance'],
        ['shot_angle'],
        ['strength_category']
    ):
        excluded = result[needed].isna().any(axis=1).sum()

        print(
            f'  {", ".join(needed)} : '
            f'{excluded} événement(s) exclu(s)'
        )

    # Extrait pour le blog
    print(
        '\nExtrait pour le blog :\n',
        result.head(10).to_string(index=False)
    )

    return result


if __name__ == '__main__':
    build_dataset()
