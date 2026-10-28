# pyCBA — Camera Beam Analyzer (Python)

Port Python de **CameraBeamAnalyzer** (MATLAB) : mesure du diamètre et du profil d'un faisceau laser
à partir d'images de caméra (ASCII/CSV exportés par Spiricon, WinCamD, BeamGage, Pyrocam, Photon…),
selon les algorithmes ISO 11146 du programme d'origine.

* Python ≥ 3.10, Windows / Linux / macOS
* Interface : PyQt5 (défaut d'Anaconda) ou PySide6, via `pyqtgraph`
* Pas d'installation de pyCBA : on lance le script depuis le dossier décompressé

## 1. Prérequis

Avec **Anaconda** (Spyder inclus), tout est déjà là sauf éventuellement `pyqtgraph` :

```
conda install -c conda-forge pyqtgraph
```

Paquets utilisés : `numpy`, `scipy`, `Pillow`, `pyqtgraph` (≥ 0.13), `PyQt5` ou `PySide6`.
Sans Anaconda : `pip install numpy scipy pillow pyqtgraph PyQt5`.

## 2. Lancer le programme

Décompressez le dossier `pyCBA` où vous voulez (ex. `C:\Users\vous\pyCBA`), puis au choix :

| Méthode | Comment |
|---|---|
| **Double-clic** (Windows) | `run_cba.bat` — cherche Anaconda, l'active, lance le programme. On peut aussi glisser une image sur le `.bat`. |
| **Spyder** | ouvrir `run_cba.py` et « Run file » (F5). La fenêtre reste ouverte, la console Spyder reste utilisable. |
| **Terminal** | `python run_cba.py [fichier_image]` |

Si `run_cba.bat` ne trouve pas Anaconda, ouvrez-le dans un éditeur et renseignez la variable `ACT`
(chemin de `Scripts\activate.bat`).

## 3. Utilisation (résumé)

1. **Open** : choisir une image ; les autres fichiers du dossier de même extension sont parcourables avec les boutons `<--` / `-->`.
   Le format (séparateur, décimale, en-tête, cellules vides d'un export ROI) est détecté automatiquement.
2. **Fond / filtre / énergie / dé-scintillement / échelle Z** : mêmes réglages que dans la version MATLAB, plus un bouton **Auto**
   à côté du champ Bkg (médiane d'un cadre de 5 % le long des bords de l'image, écrite dans le champ) ;
   le zoom est conservé lors d'un simple rafraîchissement et remis à zéro au chargement d'une image.
3. **Ouverture (ROI)** : rectangle, carré, ellipse, cercle ou polygone, avec poignées ; ouverture automatique ;
   **Energy / Area / Power** ; **ISO Beam size** (diamètres ISO sur les axes horizontal et vertical) et **ISO Beam size angle**
   (grand et petit diamètre ISO 11146 sur les axes principaux du faisceau, angle du grand axe — positif dans le sens
   anti-horaire à l'écran — et ellipticité ; toujours la définition 4σ, et l'ellipse inclinée est dessinée).
4. **Réticule** (`+`, `X`, `~`, `O`) : définit les lignes de profil ; **Profiles** ouvre la fenêtre des profils
   (données + ajustements) et enregistre les fichiers de profil à côté de l'image.
5. **Pixel killer** : bouton à bascule, un clic sur un pixel l'exclut du calcul.
6. **Help** : rappel des fonctions dans le programme.

## 4. Convertisseurs (hors interface)

`examples/convert_files.py` (à lancer sous Spyder, paramètres en tête de fichier) convertit en texte lisible par pyCBA :

* Spiricon **Pyrocam III `.RAW`**, Photon **`.PRW`**, et **captures d'écran** (TIFF/PNG) du logiciel caméra.

Les fonctions sont dans `pycba.io.converters`. Choix par défaut : facteur « Color Zoom » du RAW lu dans l'en-tête du fichier
(4 sinon), PRW en comptes bruts (la mise à l'échelle 0–255 reste une option), réticule des captures détecté par ses couleurs
hors palette (voir `matlab/CONVERTERS_FIXES.txt`).

## 5. Contenu du dossier

```
run_cba.bat / run_cba.py      lanceurs
src/pycba/io/                 lecture des fichiers (ASCII, palette, convertisseurs)
src/pycba/core/               calculs (métriques ISO, profils, ROI, mesures) — sans interface
src/pycba/gui/                interface Qt/pyqtgraph (les callbacks ne font que de la gestion d'IHM)
src/pycba/resources/          palette Spiricon_128bit_RIFF.pal
SampleBeams/                  vos images d'exemple (le bouton Open s'y ouvre par défaut)
examples/                     convert_files.py, check_beam_peak.py
tests/                        tests automatiques + fichiers d'exemple (tests/data)
matlab/                       correctifs à reporter dans le code MATLAB (voir §7)
docs/FUNCTION_MAP.md          correspondance fonction MATLAB → fonction Python
```