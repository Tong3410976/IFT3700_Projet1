
"""Milestone 1, Q3: données ordonnées des tirs cadrés et buts de la LNH."""

import json
import math
from pathlib import Path
import numpy as np

import pandas as pd
from scipy.ndimage import gaussian_filter

COLUMNS = [
    'game_id', 'event_id', 'season', 'game_type', 'game_date',
    'period', 'period_type', 'time_in_period', 'event_type', 'is_goal',
    'team_id', 'team_name', 'x_coord', 'y_coord', 'zone_code', 'shooter_id',
    'shooter_name', 'goalie_id', 'goalie_name', 'shot_type',
    'empty_net', 'strength', 'strength_category', 'situation_code',
    'shot_distance', 'shot_angle',
]


def get_target_net(team_id, home_team_id, defending_side):
    """
    Calcule la position du filet visé par le tir.

    Parameters
    ----------
    team_id : int
        Identifiant de l'équipe.
    home_team_id : int
        Identifiant de l'équipe à domicile.
    defending_side : str
        Côté défendu par l'équipe à domicile.

    Returns
    -------
    int
        Position du filet visé par le tir.
    """
    if team_id is None or home_team_id is None:
        return None

    if defending_side not in ('left', 'right'):
        return None

    if team_id == home_team_id:
        return -89 if defending_side == 'right' else 89

    return 89 if defending_side == 'right' else -89

def calculate_shot_distance(x, y, team_id, home_team_id, defending_side):
    """
    Calcule la distance entre le tir et le filet.

    Parameters
    ----------
    x : float
        Coordonnée x du tir.

    y : float
        Coordonnée y du tir.

    team_id : int
        Identifiant de l'équipe.

    home_team_id : int
        Identifiant de l'équipe à domicile.

    defending_side : str
        Côté défendu par l'équipe à domicile.

    Returns
    -------
    float
        Distance entre le tir et le filet.
    """
    if x is None or y is None:
        return None

    net_x = get_target_net(team_id, home_team_id, defending_side)

    return math.hypot(x - net_x, y) if net_x is not None else None

# Voir AI logs
def deduce_home_defending_sides(game_data):
    """
    Déduit le côté défendu par l'équipe à domicile pour chaque période
    en observant le premier tir en zone offensive (pour les saisons où
    homeTeamDefendingSide n'est pas fourni par l'API, ex. 2016 à 2019).
    """
    home_id = (game_data.get('homeTeam') or {}).get('id')
    periods = {}

    for play in game_data.get('plays', []):
        p_num = (play.get('periodDescriptor') or {}).get('number')
        if p_num is None or p_num in periods:
            continue

        det = play.get('details') or {}
        zone = det.get('zoneCode')
        x = det.get('xCoord')
        team = det.get('eventOwnerTeamId')

        if zone == 'O' and x is not None and team is not None:
            if team == home_id:
                periods[p_num] = 'left' if x > 0 else 'right'
            else:
                periods[p_num] = 'right' if x > 0 else 'left'

    opposite = {'left': 'right', 'right': 'left'}
    if 1 in periods:
        if 2 not in periods:
            periods[2] = opposite[periods[1]]
        if 3 not in periods:
            periods[3] = periods[1]
        if 4 not in periods and 3 in periods:
            periods[4] = opposite[periods[3]]
    elif 2 in periods:
        periods[1] = opposite[periods[2]]
        if 3 not in periods:
            periods[3] = periods[1]

    return periods


def calculate_shot_angle(x, y, team_id, home_team_id, defending_side):
    """
    Angle absolu  (0 à 90 degrés) entre le tir et le filet.
    Les tirs derrière la ligne de but restent inconnus.

    Parameters
    ----------
    x : float
        Coordonnée x du tir.

    y : float
        Coordonnée y du tir.

    team_id : int
        Identifiant de l'équipe.

    home_team_id : int
        Identifiant de l'équipe à domicile.

    defending_side : str
        Côté défendu par l'équipe à domicile.

    Returns
    -------
    float
        Angle entre le tir et le filet.
    """

    if x is None or y is None:
        return None

    net_x = get_target_net(team_id, home_team_id, defending_side)

    if net_x is None:
        return None

    # Ne pas attribuer  0 degré
    # aux tirs effectués derrière le filet.
    if (net_x == 89 and x > 89) or (net_x == -89 and x < -89):
        return None

    return math.degrees(math.atan2(abs(y), abs(net_x - x)))


def get_player_name(player_id, roster):
    """
    Avoir le nom du joueur

    Parameters
    ----------
    player_id : int
        Identifiant du joueur.

    roster : dict
        Dictionnaire contenant les joueurs.

    Returns
    -------
    str
        Nom du joueur.
    """
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

    # ag, av, hv, hg = gardien visiteur, patineurs visiteurs, patineurs domicile, gardien domicile
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

    game_id = game_data.get('id')
    home = game_data.get('homeTeam') or {}
    away = game_data.get('awayTeam') or {}

    home_id = home.get('id')
    away_id = away.get('id')

    roster = {
        p['playerId']: p
        for p in game_data.get('rosterSpots', [])
        if p.get('playerId') is not None
    }

    saison = game_data.get('season')
    home_sides = deduce_home_defending_sides(game_data)

    for play in game_data.get('plays', []):
        kind = play.get('typeDescKey')

        # Conserver les tirs cadrés et les buts uniquement
        if kind not in ('shot-on-goal', 'goal'):
            continue

        det = play.get('details') or {}

        event_id = play.get('eventId')

        team_id = det.get('eventOwnerTeamId')

        if kind == 'goal':
            shooter_id = det.get('scoringPlayerId')
        else:
            shooter_id = det.get('shootingPlayerId')

        goalie_id = det.get('goalieInNetId')

        x = det.get('xCoord')
        y = det.get('yCoord')

        zone_code = det.get('zoneCode')

        period = play.get('periodDescriptor') or {}
        period_number = period.get('number')

        side = play.get('homeTeamDefendingSide')
        if side is None:
            side = home_sides.get(period_number)

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

        shot_distance = calculate_shot_distance(x, y, team_id, home_id, side)
        shot_angle = calculate_shot_angle(x, y, team_id, home_id, side)
        
        rows.append({
            'game_id': game_id,
            'event_id': event_id,
            'season': saison,
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
            'zone_code' : zone_code,
            'shooter_id': shooter_id,
            'shooter_name': get_player_name(shooter_id, roster),
            'goalie_id': goalie_id,
            'goalie_name': get_player_name(goalie_id, roster),
            'shot_type': det.get('shotType'),
            'empty_net': empty_net,
            'strength': strength,
            'strength_category': category,
            'situation_code': scode,
            'shot_distance': shot_distance,
            'shot_angle': shot_angle
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



def normalize_shots(pixel_x = None,pixel_y = None):
    """
        Transforme les coordonnées de chaque tir dans le fichier csv pour les orienter dans la demi-glace.
        En option on peux convertir les coordonnées en pixel.
    """
    root = Path(__file__).resolve().parents[2]
    data_shots = pd.read_csv(root/'data'/'processed'/'shots.csv')

    # Garder toutes les colonnes ?
    '''
     columns = [
        "game_id",
        "event_id",
        "season",
        "game_type",
        "team_id",
        "period",
        "x_coord",
        "y_coord"
    ] 
    '''
    # data_shots_normalize = data_shots[columns].copy()
    data_shots_normalize = data_shots.copy()
    
    offensive_events = data_shots.loc[data_shots_normalize["zone_code"] == "O"]

    x_mean = (offensive_events.groupby(["game_id", "team_id", "period"])["x_coord"].mean().rename('x_mean'))

    data_shots_normalize = data_shots_normalize.join(x_mean, on= ['game_id', 'team_id', 'period'])
    # S'il n'y a pas eu d'événement O pendant une période alors on utilise juste la coordonnée de x de l'événement  
    invert = data_shots_normalize['x_mean'].fillna(data_shots_normalize['x_coord']) < 0

    # On rajoute les colonnes 
    data_shots_normalize["x_norm"] = data_shots_normalize["x_coord"]
    data_shots_normalize["y_norm"] = data_shots_normalize["y_coord"]
    data_shots_normalize.loc[invert, ["x_norm", "y_norm"]] *= -1

    # On a plus besoin de la colonne moyenne
    data_shots_normalize = data_shots_normalize.drop(columns=["x_mean"])

    if pixel_x is not None and pixel_y is not None:
        data_shots_normalize["x_pixel"] = (data_shots_normalize["x_norm"] * pixel_x)/100
        # On rajoute 42.5 car les y peuvent aller jusqu'à -42.5
        data_shots_normalize["y_pixel"] = ((data_shots_normalize["y_norm"] + 42.5) * pixel_y)/100

    data_shots_normalize.to_csv(root/'data'/'processed'/'shots_normalized.csv', index=False)

    return data_shots_normalize

def shots_per_hour(n_bins_x, n_bins_y, season, team=None):
    """
        team = team_id
        Retourne un tableau qui compte le nombre de tir et son taux de tirs par heure-équipe par région du terrain.
        Si l'équipe est précisé : Retourne le taux exédentaire de tirs de l'équipe par rapport à la ligue.
    """
    root = Path(__file__).resolve().parents[2]
    data_shots_normalized = pd.read_csv(root/'data'/'processed'/'shots_normalized.csv')

    shots = data_shots_normalized.loc[data_shots_normalized["season"] == season] 

    nbr_match = shots["game_id"].nunique()

    interval_x = 100/n_bins_x
    interval_y = 85/n_bins_y

    rows = []

    if team is not None:
        shots_team = shots.loc[(shots["team_id"] == team)]
        nbr_match_team = shots_team["game_id"].nunique()

        if nbr_match_team == 0:
            return None

    for x in np.arange(0, 100, interval_x):
        for y in np.arange(-42.5, 42.5, interval_y):
            x_min = x
            x_max = x + interval_x
            y_min = y
            y_max = y + interval_y

            if (x_max == 100) & (y_max == 42.5):
                shots_zone = shots.loc[(shots["x_norm"] >= x_min)&(shots["x_norm"] <= x_max)&(shots["y_norm"] >= y_min)&(shots["y_norm"] <= y_max)]
            elif(x_max == 100):
                shots_zone = shots.loc[(shots["x_norm"] >= x_min)&(shots["x_norm"] <= x_max)&(shots["y_norm"] >= y_min)&(shots["y_norm"] < y_max)]
            elif(y_max == 42.5):
                shots_zone = shots.loc[(shots["x_norm"] >= x_min)&(shots["x_norm"] < x_max)&(shots["y_norm"] >= y_min)&(shots["y_norm"] <= y_max)]
            else:
                shots_zone = shots.loc[(shots["x_norm"] >= x_min)&(shots["x_norm"] < x_max)&(shots["y_norm"] >= y_min)&(shots["y_norm"] < y_max)]

            nbr_shots = len(shots_zone)
            taux_shots = nbr_shots/(2*nbr_match)

            if team is None: 
                rows.append({
                    "Bins": f"({x_min}; {x_max}) ({y_min}; {y_max})",
                    "Nombre de tir" : nbr_shots,
                    "Taux de tirs par heure-équipe" : taux_shots
                })
            else:

                if (x_max == 100) & (y_max == 42.5):
                    shots_zone_team = shots_team.loc[(shots_team["x_norm"] >= x_min)&(shots_team["x_norm"] <= x_max)&(shots_team["y_norm"] >= y_min)&(shots_team["y_norm"] <= y_max)]
                elif(x_max == 100):
                    shots_zone_team = shots_team.loc[(shots_team["x_norm"] >= x_min)&(shots_team["x_norm"] <= x_max)&(shots_team["y_norm"] >= y_min)&(shots_team["y_norm"] < y_max)]
                elif(y_max == 42.5):
                    shots_zone_team = shots_team.loc[(shots_team["x_norm"] >= x_min)&(shots_team["x_norm"] < x_max)&(shots_team["y_norm"] >= y_min)&(shots_team["y_norm"] <= y_max)]
                else:
                    shots_zone_team = shots_team.loc[(shots_team["x_norm"] >= x_min)&(shots_team["x_norm"] < x_max)&(shots_team["y_norm"] >= y_min)&(shots_team["y_norm"] < y_max)]

                nbr_shots_team = len(shots_zone_team)
                taux_shots_team = nbr_shots_team/(nbr_match_team)
                taux_excess = taux_shots_team - taux_shots

                rows.append({
                    "Bins": f"({x_min}; {x_max}) ({y_min}; {y_max})",
                    "Nombre de tir de l'équipe" : nbr_shots_team,
                    "Taux de tirs par heure-équipe" : taux_shots_team,
                    "Taux de tirs de la ligue" : taux_shots,
                    "Taux d'excédentaire" : taux_excess
                })    

    df = pd.DataFrame(rows)
    df = df.set_index("Bins")

    return df

    # Allez voir AI_logs (j'ai retiré les commentaires inutiles)
def get_season_excess_shots(season, n_bins_x=50, n_bins_y=42, sigma=1.5):
    """
    Calcule pour une saison donnée les matrices 2D du taux excédentaire 
    de chaque équipe par zone, lissé avec un noyau gaussien.
    Retourne un dictionnaire contenant les matrices par équipe, 
    ainsi que les bordures des axes X et Y pour Plotly.
    """
    root = Path(__file__).resolve().parents[2]
    df = pd.read_csv(root / 'data' / 'processed' / 'shots_normalized.csv')
    
    # Isoler les tirs de la saison
    shots = df[(df['season'] == season)]
    
    # Calcul du taux moyen de la ligue
    n_matchs_ligue = shots['game_id'].nunique()
    
    # np.histogram2d permet de facilement identifier le nombre de tirs par zone
    ligue_counts, xedges, yedges = np.histogram2d(
        shots['x_norm'], shots['y_norm'], 
        bins=[n_bins_x, n_bins_y], 
        range=[[0, 100], [-42.5, 42.5]]
    )

    taux_ligue = ligue_counts / (2 * n_matchs_ligue)
    
    # Calcul du taux excédentaire pour chaque équipe
    team_matrices = {}
    
    # Extraire les équipes uniques pour itérer dessus
    equipes = shots[['team_id', 'team_name']].drop_duplicates().sort_values('team_name')
    
    for row in equipes.itertuples(index=False):
        t_shots = shots[shots['team_id'] == row.team_id]
        n_matchs_team = t_shots['game_id'].nunique()
        
        if n_matchs_team == 0:
            continue
            
        # Décompte des tirs pour cette équipe spécifique
        t_counts, _, _ = np.histogram2d(
            t_shots['x_norm'], t_shots['y_norm'], 
            bins=[n_bins_x, n_bins_y], 
            range=[[0, 100], [-42.5, 42.5]]
        )
        
        # Taux par match pour l'équipe
        taux_equipe = t_counts / n_matchs_team
        
        # Différence avec la moyenne de la ligue
        excedent = taux_equipe - taux_ligue
        
        # Lissage gaussien et transposition
        # La transposition (.T) est cruciale pour que la matrice soit orientée 
        # correctement (X horizontal, Y vertical).
        excedent_lisse = gaussian_filter(excedent.T, sigma=sigma)
        
        team_matrices[row.team_name] = excedent_lisse
        
    return team_matrices, xedges, yedges
