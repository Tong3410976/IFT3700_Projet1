import json
import os
from pathlib import Path

import requests


# Définir la racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dossier par défaut pour les données brutes.
# Chaque membre de l'équipe peut définir la variable d'environnement
# IFT6758_DATA_DIR pour utiliser un autre emplacement local.
DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "IFT6758_DATA_DIR",
        str(PROJECT_ROOT / "data" / "raw"),
    )
)


def build_regular_game_id(season_start: int, game_number: int) -> int:
    """
    Construire l'identifiant d'un match de saison régulière.

    Parameters
    ----------
    season_start : int
        Année de début de la saison, par exemple 2023 pour 2023-2024.

    game_number : int
        Numéro du match dans la saison régulière.

    Returns
    -------
    int
        Identifiant du match de la LNH.
    """

    # "02" correspond à un match de saison régulière
    game_id = f"{season_start}02{game_number:04d}"

    return int(game_id)


def get_game_ids(season_id: int, game_type: int) -> list[int]:
    """
    Récupérer les identifiants des matchs d'une saison donnée.

    Parameters
    ----------
    season_id : int
        Identifiant de la saison, par exemple 20232024.

    game_type : int
        Type de match :
        2 pour la saison régulière,
        3 pour les séries éliminatoires.

    Returns
    -------
    list[int]
        Liste des GAME_ID correspondant à la saison et au type de match.
    """

    # Vérifier que le type de match est valide
    if game_type not in (2, 3):
        raise ValueError(
            "game_type doit être 2 (saison régulière) "
            "ou 3 (séries éliminatoires)."
        )

    # URL permettant d'obtenir la liste des matchs
    url = "https://api.nhle.com/stats/rest/en/game"

    # Filtrer selon la saison et le type de match
    params = {
        "cayenneExp": f"season={season_id} and gameType={game_type}",
        "limit": 2000,
    }

    # Envoyer la requête
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    # Convertir la réponse JSON
    data = response.json()

    # Extraire uniquement les GAME_ID
    game_ids = [match["id"] for match in data["data"]]

    # Trier les identifiants pour avoir un ordre stable
    return sorted(game_ids)


def download_game(
    game_id: int,
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> dict:
    """
    Télécharger ou charger les données play-by-play d'un match de la LNH.

    Parameters
    ----------
    game_id : int
        Identifiant unique du match.

    data_dir : Path | str
        Dossier où les fichiers JSON sont sauvegardés.

    Returns
    -------
    dict
        Données du match sous forme de dictionnaire Python.
    """

    # Accepter aussi bien un objet Path qu'une chaîne de caractères
    data_dir = Path(data_dir)

    # Créer le dossier de données s'il n'existe pas
    data_dir.mkdir(parents=True, exist_ok=True)

    # Construire le chemin du fichier local
    file_path = data_dir / f"{game_id}.json"

    # Charger les données locales si le fichier existe déjà
    if file_path.exists():
        print("Chargement depuis le cache :", file_path.name)

        with open(file_path, "r", encoding="utf-8") as fichier:
            return json.load(fichier)

    # Sinon, télécharger les données depuis l'API
    print("Téléchargement depuis l'API :", game_id)

    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    game_data = response.json()

    # Sauvegarder les données localement
    with open(file_path, "w", encoding="utf-8") as fichier:
        json.dump(game_data, fichier, ensure_ascii=False, indent=2)

    return game_data


def download_season(
    season_id: int,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    max_games: int | None = None,
) -> None:
    """
    Télécharger les matchs de saison régulière et des séries éliminatoires
    pour une saison donnée.

    Parameters
    ----------
    season_id : int
        Identifiant de la saison, par exemple 20232024.

    data_dir : Path | str
        Dossier racine où les données de la saison seront sauvegardées.

    max_games : int | None
        Nombre maximal de matchs à télécharger pour chaque type.
        Utile pour effectuer des tests.
        Si None, tous les matchs sont téléchargés.
    """

    data_dir = Path(data_dir)

    # Associer le type numérique à un nom de dossier
    game_types = {
        2: "regular",
        3: "playoffs",
    }

    for game_type, folder_name in game_types.items():

        # Récupérer les GAME_ID réels depuis l'API
        game_ids = get_game_ids(season_id, game_type)

        # Limiter le nombre de matchs pendant les tests
        if max_games is not None:
            game_ids = game_ids[:max_games]

        # Créer un dossier propre pour cette saison et ce type de match
        season_dir = data_dir / str(season_id) / folder_name

        print()
        print(
            f"Téléchargement de {len(game_ids)} matchs "
            f"({folder_name}) pour la saison {season_id}"
        )

        # Télécharger chaque match
        for game_id in game_ids:
            try:
                download_game(game_id, data_dir=season_dir)

            except requests.HTTPError as erreur:
                if (
                    erreur.response is not None
                    and erreur.response.status_code == 404
                ):
                    print(
                        "Match ignoré (play-by-play indisponible) : "
                        f"{game_id}"
                    )
                else:
                    print(
                        f"Erreur HTTP pour le match {game_id} : "
                        f"{erreur}"
                    )

            except requests.RequestException as erreur:
                # Continuer même en cas de problème de connexion
                print(
                    f"Erreur de connexion pour le match {game_id} : "
                    f"{erreur}"
                )
