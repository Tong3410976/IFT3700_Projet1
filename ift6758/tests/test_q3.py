
from ift6758.features.shot_features import (
    extract_shots,
    parse_situation,
    calculate_shot_angle,
    calculate_shot_distance,
)


def test_standard_and_empty_net():
    assert parse_situation(
        '1551', 13, 13
    ) == ('5v5', False, 'forces égales')

    assert parse_situation(
        '1451', 13, 13
    ) == ('5v4', False, 'avantage numérique')

    assert parse_situation(
        '0651', 13, 13
    ) == ('5v6', True, None)

    assert parse_situation(
        '1010', 14, 13
    ) == (None, None, None)


def test_one_row_per_event_and_missing_data():
    game = {
        'id': 2023030111,
        'season': 20232024,
        'gameType': 3,
        'homeTeam': {
            'id': 13,
            'abbrev': 'FLA'
        },
        'awayTeam': {
            'id': 14,
            'abbrev': 'TBL'
        },
        'rosterSpots': [],
        'plays': [
            {
                'eventId': 1,
                'typeDescKey': 'shot-on-goal',
                'situationCode': '1551',
                'homeTeamDefendingSide': 'right',
                'details': {
                    'eventOwnerTeamId': 13,
                    'xCoord': -84,
                    'yCoord': 0,
                    'shootingPlayerId': 123
                }
            },
            {
                'eventId': 2,
                'typeDescKey': 'missed-shot'
            },
            {
                'eventId': 3,
                'typeDescKey': 'goal',
                'situationCode': '1010',
                'homeTeamDefendingSide': 'left',
                'details': {
                    'eventOwnerTeamId': 14,
                    'xCoord': 71,
                    'yCoord': 7
                }
            },
        ],
    }

    frame = extract_shots(game)

    assert len(frame) == 2
    assert frame['is_goal'].sum() == 1
    assert frame.iloc[0]['shot_distance'] == 5
    assert frame.iloc[0]['shooter_name'] is None
    assert frame.iloc[1]['strength'] is None
    assert frame.iloc[1]['empty_net'] is None

    assert calculate_shot_angle(
        91, 28, 14, 13, 'right'
    ) is None

def test_old_season_zone_fallback():
    assert calculate_shot_distance(78, -19, 8, 8, None, "O") is not None
    assert calculate_shot_angle(78, -19, 8, 8, None, "O") is not None
