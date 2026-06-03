# SUIVI DE PROJET

> Projet :  Édition d'images de visages par manipulation de codes latents \
> Encadrant :  Yann Gousseau

---

## S1 (4 mai)
Mise en place du projet avec le prof

## S2 (11 mai)
- **Tout le monde** | Lecture de la littérature
- **Baptiste et Antonin** | Création du git
- **Antonin en concertation avec tout le groupe** | Choisir les modèles à utiliser
- **Antonin** | Faire un `.ipynb` sur le sampling de $\mathcal{Z}$ et $\mathcal{W}$
- **Baptiste** | Gérer les problèmes de compatibilité entre le projet et Windows + RTX

## S3 (18 mai)
- **Colin** | Premiers tests avec la PCA et essais de déplacement affine par projection pour ajouter / enlever des lunette
- **Baptiste** | Premiers essais de projection image -> $\mathcal{Z}$ / $\mathcal{W}$ par optimisation, avec:
  - Une première boucle qui choisis une meilleur point de départ parmis des points samplés aléatoirement
  - Une deuxième boucle d'optimisation en utilisant
```math
ℒ = ℒ_{pixel} + ℒ_{features} = MSE(pixels_{pred}, pixels_{target}) + MSE(features_{pred}, features_{target})
```
  - Résultats faibles même si parfois chanceux.
- **Antonin** | Demonstration de la classification pour la reproduction de InterFaceGAN
  - Utilisation d'un ResNET50 pre-entrainer (poids de PyTorch) avec les classes `sunglass` et `sunglasses`
  - Résultats corrects, mais le pas est très variable en fonction des images et la direction est très liée à l'age (et dans une moindre mesure, le genre).

## S4 (25 mai)
- **Colin** | Un essai intéressant de fine-tuning de resnet pour trouver w à partir d'une image
- **Baptiste** | Relecture plus en profondeur de StyleGan, InterfaceGan, StyleGan2, Analyzing and Improving the Image Quality of StyleGAN
- **Antonin** | Génération d'un jeu de données et fine tunning d'un classifier
  - Dataset de vecteurs dans $\mathcal{Z}$ et dans $\mathcal{W}$ et d'images associées (~65k)
  - Fine tunning d'un ResNET18 sur CelebA pour la classification de 3 attributs (`eyewear`, `male`, `young`)
  - Tentative de "deliage" lunettes/age (formule d'InterFaceGAN), résultats relativement satisfaisant mais une amélioration plutôt faible

## S5 (1 juin)
- **Colin** | Avancement sur le fine-tuning du resnet, résultats prometteurs
- **Baptiste** | Quasi finalisation de la projection par optimisation:
  - Centrer les visages avec landmarks 
  - Première boucle de sampling aléatoire sur 1500 éléments pour compute le centre de W, sa variance selon chaque composante, et un point de départ à priori bon 
  - Optimisation directement sur W
  - Optimisation également sur les layers de bruit de styleGAN2
  - Diffusion (ajout de bruit au vecteur latent à chaque iteration)
  - Loss utilisée ($ℒ_{RegNoise}$ étant la loss décrite dans "Analyzing and Improving the Image Quality of StyleGAN") :
```math
ℒ = 4 * ℒ_{pixel} + 0.5 * ℒ_{LPIPS} + 2e4 * ℒ_{RegNoise}
```
  - Résultats très bons, temps < 5min
- **Antonin** | Compression du dataset et Tentative de projection apprise
  - Utilisaton d'image en uint8 et de vecteurs en float16 (dataset 84Go -> 10Go)
  - Fine tunning d'un ResNet18 puis ResNet34 pour predire le vecteur dans $\mathcal{W}$ associé au visage
    - Avec simplement une loss MSE sur $\mathcal{W}$, la prediction fonctionne relativement bien pour des visages générés mais beaucoup moins bien pour de vraies images
    - En ajoutant un terme de reconstruction, l'entrainement devient trop lent pour avoir une longeur raisonnable

- **Quentin** | Avancement sur la PCA et Style Mixing
    - Analyse de la PCA sur 1000 échantillons de l'espace W,
    - Début du désenchevêtrement des composantes comme présenté dans le papier numéro 3
    - Essais de style mixing en combinant deux codes latents.

## S6 (8 juin)
...

## S7 (15 juin)
...

## S8 (22 juin)
...
