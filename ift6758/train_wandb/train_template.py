# Ce fichier est un template que nous pourrons utiliser pour enregistrer une 
# expérience dans le projet wandb. En suivant ce format, nous nous assurons que 
# le format sera uniforme pour toutes les expériences. Ici l'exemple sera une 
# simple régression logistique sur les distances de tirs. Pour en savoir plus 
# sur les choses que l'on peut faire, je vous invite à consulter : https://www.youtube.com/watch?v=hmewPDNUNJs&list=PLD80i8An1OEGajeVo15ohAQYF1Ttle0lk et la documentation officielle https://docs.coreweave.com/forge-home c'est là que je 
# suis allé pour faire le template.


from pathlib import Path


import wandb
# Devrais vous demander une clé API qu'une seule fois, par après ça ira 
# chercher votre clé personelle dans vos fichiers.
wandb.login()
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

# On définit les variables ici pour éviter de faire des erreurs entre la 
# description que l'on donne au run Wandb et le modèle que nous entraînons

# Config est l'identité de l'expérience qui va être enregistrer dans notre run # wandb. C'est un dictionnaire python, pour les autres modèles nous sommes 
# libre d'ajouter et de retirer des informations, le but est que la config 
# donne tous les détails de l'expérience faite. Dans ce cas, "features": 
# ["shot_distance"] sera la liste d'entrée en X utilisé pour l'entraînement,  
# "target": "is_goal" est le nom de la colonne cible. Le "random_state": 42 est 
# important pour la reproductibilité etc...
CONFIG = {
    "model_type": "LogisticRegression",
    "features": ["shot_distance"],
    "target": "is_goal",
    "random_state": 42,
    "max_iter": 1000,
}
# Les tags sont utiles pour se retrouver dans le projet, ils permettent 
# d'étiqueter les expériences et permet ensuite de les filtrer selon les tags
TAGS = ["template", "logistic_regression"]
# Nom du run
EXPERIMENT_NAME = "template_exemple_logistic_regression"
# Important sinon le run n'est pas à la bonne place
ENTITY = "felix-carignan-universite-de-montreal"

def main():
    # Généralement on appelle cette fonction une fois par script pour créer un # "run".

    run = wandb.init(
            entity=ENTITY,
            project="IFT3700-2026-04",
            name=EXPERIMENT_NAME,
            config=CONFIG,
            tags=TAGS
    )

    # Pour avoir accès à l'url de l'expérience on peut simplement faire ceci :
    run_url = run.url
    print(f"Voici l'url de l'expérience : {run_url}")

    # Une fois que nous avons initié le run, on fait run le modèle comme 
    # normalement (pour le template je ne ferais pas de validation croisée ni 
    # de séparation de données, mais pour les vrais expériences il est 
    # important que tout soit fait correctement): 

    # Définition de la racine du projet
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed" / "shots.csv"
    df = pd.read_csv(data_path)

    df = df.dropna(subset=CONFIG["features"] + [CONFIG["target"]])
    X_train = df[CONFIG["features"]]
    Y = df['is_goal']

    # Ici il est important que les hyperparamètres de notre modèle respecte ce # que nous avons mis dans la variable CONFIG
    model = LogisticRegression(
        max_iter=CONFIG["max_iter"],
        random_state=CONFIG["random_state"],
    )
    model.fit(X_train, Y)

    # Avec chaque run on peut log différentes métriques dans le dashboard, ici 
    # je vais log la courbe AUC et la log_loss pour nous donner une idée 
    # comment tracer certaines courbes.
    # Ici c'est terrible de tester sur l'ensemble d'entraînement mais c'est 
    # seulement un exemple.
    y_pred = model.predict(X_train)
    y_proba = model.predict_proba(X_train)[:,1]
    
    # Aller chercher les métriques que l'on souhaite afficher
    y_accuracy = accuracy_score(Y, y_pred)
    y_auc = roc_auc_score(Y, y_proba)
    y_loss = log_loss(Y, y_proba)
    
    # C'est ce qui sera dans le tableau de bord de Wandb
    run.log({
        "y_accuracy": y_accuracy,
        "y_roc_auc": y_auc,
        "y_loss": y_loss,
    })

    # Maintenant on passe à l'étape de sauvegarder et enregistrer le modèle, 
    # pour comprendre un peu mieux comment tout fonctionne, la documentation 
    # l'explique très bien : https://docs.coreweave.com/products/wandb/artifacts/construct-an-artifact

    # On créer le dossier de modèles et créer le chemin pour notre run, joblib # nous permet de sérialiser notre modèle sklearn (https://scikit-learn.org/stable/model_persistence.html)

    models_dir = project_root/"models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir/f"{EXPERIMENT_NAME}.joblib"

    joblib.dump(model, model_path)

    artifact = wandb.Artifact(
        name=f"model_{EXPERIMENT_NAME}",
        type="model",
        description="Template de fichier wandb test",
        metadata=CONFIG,
    )
    # On ajoute le fichier du modèle sérialiser 
    artifact.add_file(str(model_path))
    # log_artifact envoie le run sur wandb mais pour rester organiser on doit aussi le lier à un registre, à voir comment gérer les registres mais voici un exemple :
    run.log_artifact(artifact)
    run.link_artifact(
        artifact=artifact,
        target_path=f"{run.project}/test_models"
    )
    run.finish()

if __name__ == "__main__":
    main()