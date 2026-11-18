from pathlib import Path
import wandb
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

# La structure de ce fichier est inspiré de train_template.py, ce fichier va 
# s'occuper d'entraîner et enregistrer 3 modèles différents de régression 
# logistique.

# Les configs de nos 3 modèles
CONFIG_DIST = {
    "model_type": "LogisticRegression",
    "features": ["shot_distance"],
    "target": "is_goal",
    "random_state": 42,
}
CONFIG_ANGLE = {
    "model_type": "LogisticRegression",
    "features": ["shot_angle"],
    "target": "is_goal",
    "random_state": 42,
}
CONFIG_DIST_ANGLE = {
    "model_type": "LogisticRegression",
    "features": ["shot_distance", "shot_angle"],
    "target": "is_goal",
    "random_state": 42,
}

# Les tags
TAGS_DIST = ["shot_distance", "logistic_regression", "base_models"]
TAGS_ANGLE = ["shot_angle", "logistic_regression", "base_models"]
TAGS_DIST_ANGLE = ["shot_distance","shot_angle", "logistic_regression", "base_models"]
# Important sinon le run n'est pas à la bonne place
ENTITY = "felix-carignan-universite-de-montreal"
PROJECT = "IFT3700-2026-04"

# Chemins des donnees
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_set.csv"
DATA_VAL_PATH = PROJECT_ROOT / "data" / "processed" / "validation_set.csv"
MODELS_DIR = PROJECT_ROOT / "models"

# Liste des 3 expériences
model_dist = {
    "name": "logreg_distance",
    "config": CONFIG_DIST,
    "tags": TAGS_DIST,
    "description": "Régression logistique simple sur shot_distance"
}
model_angle = {
    "name": "logreg_angle",
    "config": CONFIG_ANGLE,
    "tags": TAGS_ANGLE,
    "description": "Régression logistique simple sur shot_angle"
}
model_dist_angle = {
    "name": "logreg_distance_angle",
    "config": CONFIG_DIST_ANGLE,
    "tags": TAGS_DIST_ANGLE,
    "description": "Régression logistique simple sur shot_distance et shot_angle"
}

MODELS = [model_dist, model_angle, model_dist_angle]

# Cette fonction s'occupe d'entraîner un modèle de régression logistique 
# simple, puis d'enregistrer l'expérience et ces métriques sur wandb
def train_single_log_reg(exp_info, train_df, val_df):
    # Initiation du run
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=exp_info["name"],
        config=exp_info["config"],
        tags=exp_info["tags"]
    )
    # Afficher le lien
    run_url = run.url
    print(f"Voici l'url de l'expérience {exp_info['name']} : {run_url}")
    # Preparer les donnees
    X_train = train_df[exp_info["config"]["features"]]
    Y_train = train_df['is_goal']
    X_val = val_df[exp_info["config"]["features"]]
    Y_val = val_df['is_goal']
    # Declaration du modele
    model = LogisticRegression(random_state=exp_info["config"]["random_state"])
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_val)
    Y_score = model.predict_proba(X_val)[:,1]

    # Pour avoir plus de details sur chaque modele on va afficher ces graphiques
    Y_accuracy = accuracy_score(Y_val, Y_pred)
    Y_auc = roc_auc_score(Y_val, Y_score)
    Y_loss = log_loss(Y_val, Y_score)
    
    # C'est ce qui sera dans le tableau de bord de Wandb
    run.log({
        "y_accuracy": Y_accuracy,
        "y_roc_auc": Y_auc,
        "y_loss": Y_loss,
    })

    # Toute la logique d'enregistrement de l'expérience sur wandb
    models_dir = PROJECT_ROOT/"models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir/f"{exp_info['name']}.joblib"

    joblib.dump(model, model_path)

    artifact = wandb.Artifact(
        name=f"model_{exp_info['name']}",
        type="model",
        description=exp_info["description"],
        metadata=exp_info["config"],
    )
    # On ajoute le fichier du modèle sérialiser 
    artifact.add_file(str(model_path))
    # log_artifact envoie le run sur wandb mais pour rester organiser on doit aussi le lier à un registre, à voir comment gérer les registres mais voici un exemple :
    run.log_artifact(artifact)
    run.link_artifact(
        artifact=artifact,
        target_path=f"{run.project}/logistic_regression_models"
    )
    run.finish()
    return run_url

def main():
    wandb.login()
    # Extraction des ensembles 
    train_all = pd.read_csv(DATA_TRAIN_PATH)
    val_all = pd.read_csv(DATA_VAL_PATH)

    # On retire les nan
    train_clean = train_all.dropna(subset=["shot_angle", "shot_distance", "is_goal"])
    val_clean = val_all.dropna(subset=["shot_angle", "shot_distance", "is_goal"])

    # On entraine les 3 modèles
    urls = {}
    for model in MODELS:
        urls[model['name']] = train_single_log_reg(model, train_clean, val_clean)
    
    # Afficher les liens 
    for name, url in urls.items():
        print(f"- {name} : {url}")
    
if __name__ == "__main__":
    main()