
"""Milestone 1, Q3: données ordonnées des tirs cadrés et buts de la LNH."""

import json
import math
from pathlib import Path

import pandas as pd


COLUMNS = [
    'game_id', 'event_id', 'season', 'game_type', 'game_date',
    'period', 'period_type', 'time_in_period', 'event_type', 'is_goal',
    'team_id', 'team_name', 'x_coord', 'y_coord', 'shooter_id',
    'shooter_name', 'goalie_id', 'goalie_name', 'shot_type',
    'empty_net', 'strength', 'strength_category', 'situation_code',
    'shot_distance', 'shot_angle',
]


def get_target_net(team_id, home_team_id, defending_side):
    if team_id is None or home_team_id is None:
        return None

    if defending_side not in ('left', 'right'):
        return None

    if team_id == home_team_id:
        return -89 if defending_side == 'right' else 89

    return 89 if defending_side == 'right' else -89


def calculate_shot_distance(x, y, team_id, home_team_id, defending_side):
    if x is None or y is None:
        return None

    net_x = get_target_net(team_id, home_team_id, defending_side)

    return math.hypot(x - net_x, y) if net_x is not None else None


def calculate_shot_angle(x, y, team_id, home_team_id, defending_side):
    """
    Angle absolu exploratoire (0 à 90 degrés).
    Les tirs derrière la ligne de but restent inconnus.
    """

    if x is None or y is None:
        return None

    net_x = get_target_net(team_id, home_team_id, defending_side)

    if net_x is None:
        return None

    # Ne pas attribuer artificiellement 0 degré
    # aux tirs effectués derrière le filet.
    if (net_x == 89 and x > 89) or (net_x == -89 and x < -89):
        return None

    return math.degrees(math.atan2(abs(y), abs(net_x - x)))


def get_player_name(player_id, roster):
    player = roster.get(player_id) if player_id is not None else None

    if not player:
        return None

    first = (player.get('firstName') or {}).get('default', '')
    last = (player.get('lastName') or {}).get('default', '')

    return f'{first} {last}'.strip() or None


def parse_situation(code, team_id, home_team_id):
    """
    Extraire la force des équipes et détecter un filet désert.

    Code :
    - Gardien visiteur
    - Patineurs visiteurs
    - Patineurs domicile
    - Gardien domicile

    Retour :
    (strength, empty_net, strength_category)
    """

    if not isinstance(code, str) or len(code) != 4 or not code.isdigit():
        return None, None, None

    if team_id is None or home_team_id is None:
        return None, None, None

    ag, av, hv, hg = map(int, code)

    if ag not in (0, 1) or hg not in (0, 1):
        return None, None, None

    # Cas non plausibles en jeu courant
    if not (3 <= av <= 6 and 3 <= hv <= 6):
        return None, None, None

    if team_id == home_team_id:
        attack = hv
        defense = av
        defending_goalie = ag
    else:
        attack = av
        defense = hv
        defending_goalie = hg

    empty_net = defending_goalie == 0

    strength = f'{attack}v{defense}'

    # Ne pas confondre retrait de gardien et pénalité
    if ag == 0 or hg == 0:
        category = None
    elif attack == defense:
        category = 'forces égales'
    elif attack > defense:
        category = 'avantage numérique'
    else:
        category = 'désavantage numérique'

    return strength, empty_net, category


def extract_shots(game_data):
    """
    Transformer les tirs cadrés et les buts d'un match
    en DataFrame Pandas.

    Une ligne par événement.
    Identifiant unique : (game_id, event_id).
    """

    rows = []

    home = game_data.get('homeTeam') or {}
    away = game_data.get('awayTeam') or {}

    home_id = home.get('id')
    away_id = away.get('id')

    roster = {
        p['playerId']: p
        for p in game_data.get('rosterSpots', [])
        if p.get('playerId') is not None
    }

    for play in game_data.get('plays', []):
        kind = play.get('typeDescKey')

        # Conserver les tirs cadrés et les buts uniquement
        if kind not in ('shot-on-goal', 'goal'):
            continue

        det = play.get('details') or {}

        team_id = det.get('eventOwnerTeamId')

        if kind == 'goal':
            shooter_id = det.get('scoringPlayerId')
        else:
            shooter_id = det.get('shootingPlayerId')

        goalie_id = det.get('goalieInNetId')

        x = det.get('xCoord')
        y = det.get('yCoord')

        side = play.get('homeTeamDefendingSide')
        scode = play.get('situationCode')

        strength, empty_net, category = parse_situation(
            scode, team_id, home_id
        )

        if team_id == home_id:
            team_info = home
        elif team_id == away_id:
            team_info = away
        else:
            team_info = {}

        period = play.get('periodDescriptor') or {}

        rows.append({
            'game_id': game_data.get('id'),
            'event_id': play.get('eventId'),
            'season': game_data.get('season'),
            'game_type': game_data.get('gameType'),
            'game_date': game_data.get('gameDate'),
            'period': period.get('number'),
            'period_type': period.get('periodType'),
            'time_in_period': play.get('timeInPeriod'),
            'event_type': kind,
            'is_goal': int(kind == 'goal'),
            'team_id': team_id,
            'team_name': team_info.get('abbrev'),
            'x_coord': x,
            'y_coord': y,
            'shooter_id': shooter_id,
            'shooter_name': get_player_name(shooter_id, roster),
            'goalie_id': goalie_id,
            'goalie_name': get_player_name(goalie_id, roster),
            'shot_type': det.get('shotType'),
            'empty_net': empty_net,
            'strength': strength,
            'strength_category': category,
            'situation_code': scode,
            'shot_distance': calculate_shot_distance(
                x, y, team_id, home_id, side
            ),
            'shot_angle': calculate_shot_angle(
                x, y, team_id, home_id, side
            ),
        })

    return pd.DataFrame(rows, columns=COLUMNS)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]

    files = sorted((root / 'data' / 'raw').rglob('*.json'))

    if not files:
        raise FileNotFoundError('Aucun JSON sous data/raw')

    with files[0].open(encoding='utf-8') as f:
        frame = extract_shots(json.load(f))

    print(frame.head(10).to_string(index=False))

    print(
        '\nÉvénements :', len(frame),
        'Buts :', frame['is_goal'].sum()
    )

    print('\nManquants :\n', frame.isna().sum().to_string())
