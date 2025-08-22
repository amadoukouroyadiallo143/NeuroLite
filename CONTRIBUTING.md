# Guide de Contribution pour NeuroLite

Nous sommes ravis que vous souhaitiez contribuer à NeuroLite ! Ce guide vous aidera à démarrer.

## Comment Contribuer

Les contributions peuvent prendre plusieurs formes :

*   **Rapports de bugs** : Si vous trouvez un bug, veuillez ouvrir une [issue](https://github.com/amadoukouroyadiallo143/NeuroLite/issues) en utilisant le modèle de rapport de bug.
*   **Suggestions de fonctionnalités** : Vous avez une idée pour améliorer NeuroLite ? Ouvrez une [issue](https://github.com/amadoukouroyadiallo143/NeuroLite/issues) avec le modèle de demande de fonctionnalité.
*   **Code** : Corrections de bugs, nouvelles fonctionnalités, ou améliorations de la documentation.
*   **Documentation** : Améliorations du README, des docstrings, ou création de tutoriels.

## Processus de Développement

1.  **Forkez le projet** : Cliquez sur le bouton "Fork" en haut à droite de la page du dépôt.
2.  **Clonez votre fork** :
    ```bash
    git clone https://github.com/VOTRE_NOM_UTILISATEUR/NeuroLite.git
    cd NeuroLite
    ```
3.  **Créez une branche** pour votre fonctionnalité ou votre correctif :
    ```bash
    git checkout -b feature/nom-de-la-fonctionnalite
    # ou
    git checkout -b fix/description-du-bug
    ```
4.  **Installez les dépendances de développement** :
    ```bash
    pip install -r requirements-dev.txt # Nous ajouterons ce fichier plus tard
    ```
5.  **Codez !** Assurez-vous de suivre les normes de style.
6.  **Formatez votre code** avec Black avant de commiter :
    ```bash
    black .
    ```
7.  **Commitez vos changements** avec un message clair et concis :
    ```bash
    git commit -m "feat: Ajout de la fonctionnalité X"
    # ou
    git commit -m "fix: Correction du bug Y"
    ```
8.  **Poussez vos changements** vers votre fork :
    ```bash
    git push origin feature/nom-de-la-fonctionnalite
    ```
9.  **Ouvrez une Pull Request** : Allez sur le dépôt original et ouvrez une Pull Request. Remplissez le modèle de PR avec les détails de vos changements.

## Normes de Codage

*   **Style** : Nous utilisons [Black](https://github.com/psf/black) pour le formatage du code. C'est non négociable.
*   **Docstrings** : Utilisez le style de docstring de Google. Toutes les fonctions publiques doivent avoir des docstrings.
*   **Tests** : Chaque nouvelle fonctionnalité ou correction de bug doit être accompagnée de tests unitaires.

## Code de Conduite

En contribuant à ce projet, vous acceptez de respecter notre [Code de Conduite](CODE_OF_CONDUCT.md).
