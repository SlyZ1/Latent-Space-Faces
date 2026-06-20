# Code final du projet 19, Groupe 1

- `attributs-latents.ipynb`: Ce notebook montre l'édition d'attributs tels que les lunettes ou l'âge en utilisant la méthode décrite dans l'article _InterFaceGAN_. \
    Il utilise les frontières pré-calculées, car leur calcul nécessite un classifieur (ResNet18 fine-tuné) et un très large dataset (65k images).

- `projection-apprentissage.ipynb`: TODO

- `projection-perceptuelle.ipynb`: Pour améliorer les résultats du notebook précédent (`projection-apprentissage.ipynb`), nous avons exploré les idées de l'article _Encoding In Style_ (ou _pixel2style2pixel_, ou pSp). Ces idées sont décrites et partiellement implémentées (seule la loss a été utilisée, leur modèle étant considérablement plus grand que ce que notre capacité de calcul nous permet d'utiliser) dans ce notebook.
