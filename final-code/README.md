# Code final du projet 19, Groupe 1

- `attributs-latents.ipynb`: Ce notebook montre l'édition d'attributs tels que les lunettes ou l'âge en utilisant la méthode décrite dans l'article _InterFaceGAN_. \
    Il utilise les frontières pré-calculées, car leur calcul nécessite un classifieur (ResNet18 fine-tuné) et un très large dataset (65k images).

- `projection-apprentissage.ipynb`: TODO

- `projection-perceptuelle.ipynb`: Pour améliorer les résultats du notebook précédent (`projection-apprentissage.ipynb`), nous avons exploré les idées de l'article _Encoding In Style_ (ou _pixel2style2pixel_, ou pSp). Ces idées sont décrites et partiellement implémentées (seule la loss a été utilisée, leur modèle étant considérablement plus grand que ce que notre capacité de calcul nous permet d'utiliser) dans ce notebook.

- `pca_and_layer-wise_editing.ipynb` : Ce notebook est fait en deux parties. Une première sur la PCA pour explorer les directions sémantiques les plus riches. La deuxième partie traite du désenchevêtrement des attributs en appliquant des modofications de style uniquemement à certaines couches du modèle. Cette méthode est inspiré de l'article _GANSpace_. Elle est appliquée dans le notebook à la direction des lunettes obtenu dans le notebook `attributs-latents.ipynb`, mais peut être facilement adaptée pour les désenchevêtrement des attributs dans les directions obtenues avec la PCA.