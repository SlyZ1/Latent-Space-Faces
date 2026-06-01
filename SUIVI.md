# SUIVI DE PROJET

> Projet :  Édition d'images de visages par manipulation de codes latents \
> Encadrant :  Yann Gousseau

---

## S1 (4 mai)
Mise en place du projet avec le prof

## S2 (11 mai)
- Lecture de la littérature | *(Tout le monde)*
- Création du git | *(Baptiste et Antonin)*
- Choisir les modèles à utiliser | *(Antonin en concertation avec tout le groupe)*
- Faire un `.ipynb` sur le sampling de $\mathcal{W}$ et $\mathcal{W}$ | *(Antonin)*
- Gérer les problèmes de compatibilité entre le projet et Windows + RTX | *(Baptiste)*

## S3 (18 mai)
- Premiers tests avec la PCA et essais de déplacement affine par projection pour ajouter / enlever des lunettes | *(Colin)*
- Premiers essais de projection image -> $\mathcal{Z}$ / $\mathcal{W}$ par optimisation, avec:
  - Une première boucle qui choisis une meilleur point de départ parmis des points samplés aléatoirement
  - Une deuxième boucle d'optimisation en utilisant
```math
ℒ = ℒ_{pixel} + ℒ_{features} = MSE(pixels_{pred}, pixels_{target}) + MSE(features_{pred}, features_{target})
```
Résultats faibles même si parfois chanceux. | *(Baptiste)*


## S4 (25 mai)
- Un essai intéressant de fine-tuning de resnet pour trouver w à partir d'une image | *(Colin)*
- Relecture plus en profondeur de StyleGan, InterfaceGan, StyleGan2, Analyzing and Improving the Image Quality of StyleGAN | *(Baptiste)* 

## S5 (1 juin)
- Avancement sur le fine-tuning du resnet, résultats prometteurs | *(Colin)*
- Quasi finalisation de la projection par optimisation:
  - Première boucle de sampling aléatoire sur 1500 éléments pour compute le centre de W, sa variance selon chaque composante, et un point de départ à priori bon 
  - Optimisation directement sur W
  - Optimisation également sur les layers de bruit de styleGAN2
  - Diffusion (ajout de bruit au vecteur latent à chaque iteration)
  - Loss utilisée ($ℒ_{RegNoise}$ étant la loss décrite dans "Analyzing and Improving the Image Quality of StyleGAN") :
```math
ℒ = 4 * ℒ_{pixel} + 0.5 * ℒ_{features} + 1e5 * ℒ_{RegNoise}
```
  Résultats très bons, temps < 5min | *(Baptiste)*

## S6 (8 juin)
...

## S7 (15 juin)
...

## S8 (22 juin)
...
