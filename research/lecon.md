Bonne question — et ton intuition est saine 👍

Non, il n’y a volontairement pas de MaxPooling dans cette architecture.
La réduction de résolution est faite uniquement par les convolutions avec stride = 2.

Pourquoi pas de MaxPool ici ?

Dans les autoencodeurs convolutionnels (surtout pour reconstruction d’images), on évite souvent le MaxPooling pour plusieurs raisons :

1. Le stride apprend, le MaxPool jette

Conv stride 2 → la réduction spatiale est apprise

MaxPool → information perdue de façon non paramétrique

Pour un AE, perdre de l’info = reconstruction plus floue ou moins fidèle.

2. Symétrie encoder ↔ decoder

Regarde ta table :

Réduction	Méthode
128 → 64	Conv 4×4, stride 2
64 → 32	Conv 4×4, stride 2
32 → 16	Conv 4×4, stride 2
16 → 8	Conv 4×4, stride 2

Dans le decoder :

ConvTranspose stride 2 inverse exactement cette opération
👉 Avec MaxPool, il faudrait des Unpooling indices (plus complexe et rarement utilisé aujourd’hui).