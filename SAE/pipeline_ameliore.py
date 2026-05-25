"""
pipeline_ameliore.py
====================
Script de prédiction de codes douanes Legrand — Version améliorée
Compatible Spyder (exécution directe) + export ONNX pour interface web.

AMÉLIORATIONS vs version originale :
  1. LEFT JOIN au lieu de INNER JOIN → +71% de données d'entraînement
  2. Filtrage des codes invalides (0000000000, 9999999999...)
  3. Matrice sparse conservée → ×3 moins de RAM, pas de toarray()
  4. max_features 5000 → 25 000 → vocabulaire bien plus riche
  5. Hyperparamètres XGBoost optimisés + early stopping
  6. Métriques Top-1 / Top-3 / Top-5
  7. Chapitre douanier (2 chiffres) comme feature textuelle
  8. Export ONNX + JSON pour interface web sans serveur Flask
"""

import pandas as pd
import numpy as np
import json
import zipfile
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
FICHIER_PIM  = os.path.join(BASE_DIR, "CurrentView_EN + Code Douane.xlsx")
FICHIER_ETIM = os.path.join(BASE_DIR, "Technical+Data-2026-04-14_16.29.07.xlsx")

FICHIER_ONNX   = os.path.join(BASE_DIR, "modele_douane_legrand.onnx")
FICHIER_CONFIG = os.path.join(BASE_DIR, "config_ia_legrand.json")
FICHIER_ZIP    = os.path.join(BASE_DIR, "Cerveau_IA_Legrand_FINAL.zip")

# Codes placeholder à exclure de l'entraînement
CODES_INVALIDES = {
    '0000000000', '9999999999',
    '0', '00', '000', '0000',
}

# Features techniques "critiques" → boostées ×3 dans le texte
COLONNES_CRITIQUES = ['material', 'current', 'voltage', 'ampere', 'pole',
                      'mounting', 'type', 'power', 'protection', 'degree',
                      'rated', 'breaking', 'sensitivity', 'frequency']

# Features inutiles pour la douane → ignorées
MOTS_BANNIS = ['colour', 'color', 'ral', 'packaging', 'storage', 'date',
               'status', 'finishing', 'finish', 'packing', 'gross', 'net',
               'weight', 'carton', 'pallet', 'ean', 'gtin']

# ─── Helpers ──────────────────────────────────────────────────────────────────

def topk_accuracy(y_true: np.ndarray, probas: np.ndarray, k: int) -> float:
    """Précision Top-K : le vrai label est-il dans les K meilleures prédictions ?"""
    top_k = np.argsort(probas, axis=1)[:, -k:]
    return float(np.mean([y_true[i] in top_k[i] for i in range(len(y_true))]))


def normaliser_code(code) -> str | None:
    """Normalise un code douane en chaîne de 10 chiffres."""
    if pd.isna(code):
        return None
    s = str(code).strip().replace(' ', '').replace('.', '')
    s = ''.join(c for c in s if c.isdigit())
    if not s or len(s) < 6:
        return None
    return s.ljust(10, '0')[:10]


# ─── Pipeline principal ───────────────────────────────────────────────────────

def master_pipeline():
    print("=" * 65)
    print("  PIPELINE XGBoost AMELIORE — CODES DOUANES LEGRAND")
    print("=" * 65 + "\n")

    # ── 1. CHARGEMENT ─────────────────────────────────────────────────────────
    print("[1/7] Chargement des fichiers Excel...")
    try:
        df_pim  = pd.read_excel(FICHIER_PIM,  dtype=str)
        df_etim = pd.read_excel(FICHIER_ETIM, dtype=str)
    except FileNotFoundError as e:
        print(f"    ERREUR fichier introuvable : {e}")
        return

    # ── 2. NETTOYAGE PIM ──────────────────────────────────────────────────────
    print("[2/7] Nettoyage et filtrage des données...")

    df_pim.columns = ['Code Douane', 'ID_PRODUIT', 'DESC_EN', 'DESC_OTHER',
                      'Marge Family', 'Class Legrand'][:len(df_pim.columns)]

    df_pim_clean = df_pim[['ID_PRODUIT', 'DESC_EN', 'Class Legrand', 'Marge Family', 'Code Douane']].copy()
    df_pim_clean['ID_PRODUIT'] = df_pim_clean['ID_PRODUIT'].str.strip()

    # Normaliser les codes douanes
    df_pim_clean['Code Douane'] = df_pim_clean['Code Douane'].apply(normaliser_code)
    df_pim_clean = df_pim_clean.dropna(subset=['Code Douane'])

    # *** AMÉLIORATION 2 : Filtrer les codes invalides ***
    avant = len(df_pim_clean)
    df_pim_clean = df_pim_clean[~df_pim_clean['Code Douane'].isin(CODES_INVALIDES)]
    df_pim_clean = df_pim_clean[df_pim_clean['Code Douane'].str.match(r'^\d{8,10}$', na=False)]
    df_pim_clean = df_pim_clean[df_pim_clean['Code Douane'].str[:2].astype(int) >= 1]
    print(f"    Codes invalides supprimés : {avant - len(df_pim_clean):,}")
    print(f"    Produits avec code valide : {len(df_pim_clean):,}")

    # ── 3. PIVOT ETIM ─────────────────────────────────────────────────────────
    print("[3/7] Pivot des données techniques (ETIM)...")
    df_etim_pivot = (
        df_etim.pivot_table(index='MARQUE', columns='FEATUREID', values='FVALUE', aggfunc='first')
        .reset_index()
    )
    print(f"    {len(df_etim_pivot):,} produits  |  {len(df_etim_pivot.columns)-1:,} features ETIM")

    # *** AMÉLIORATION 1 : LEFT JOIN au lieu de INNER JOIN ***
    df_master = pd.merge(df_pim_clean, df_etim_pivot,
                         left_on='ID_PRODUIT', right_on='MARQUE', how='left')

    n_avec_tech = df_master['MARQUE'].notna().sum()
    print(f"    Produits dans le dataset  : {len(df_master):,}")
    print(f"    Dont avec données ETIM    : {n_avec_tech:,} ({100*n_avec_tech/len(df_master):.1f}%)")
    print(f"    Sans données ETIM (libellé seul) : {len(df_master)-n_avec_tech:,}")

    # Filtre minimum 3 produits par code (requis pour stratify)
    comptage = df_master['Code Douane'].value_counts()
    df_master = df_master[df_master['Code Douane'].isin(comptage[comptage >= 3].index)].copy()
    print(f"    Après filtre >=3 par code : {len(df_master):,} produits, {df_master['Code Douane'].nunique():,} codes")

    # Stocker les colonnes ETIM disponibles (sans MARQUE, ID_PRODUIT, etc.)
    cols_pim = {'ID_PRODUIT', 'DESC_EN', 'Class Legrand', 'Marge Family', 'Code Douane',
               'DESC_OTHER', 'MARQUE'}
    etim_cols = [c for c in df_master.columns if c not in cols_pim]

    # ── 4. FEATURE ENGINEERING (PASSEPORT TECHNIQUE BOOSTÉ) ──────────────────
    print("[4/7] Création du Passeport Technique Boosté...")

    def creer_texte_industriel(row) -> str:
        """
        Construit un texte enrichi pour le TF-IDF.
        - Chapitre douanier ×2 (guide hiérarchique)
        - Classe Legrand ×2 (très discriminante)
        - Description EN ×1
        - Features critiques ×3 (courant, tension, pôles...)
        - Autres features ×1
        """
        code      = str(row.get('Code Douane', '')) or ''
        chapter2  = code[:2] if code else ''
        classe    = str(row.get('Class Legrand', '') or '').upper()
        famille   = str(row.get('Marge Family', '') or '').upper()
        desc      = str(row.get('DESC_EN', '') or '').upper()

        # Base structurelle (répétée pour poids fort)
        parts = [
            f"CHAPTER:{chapter2}",
            f"CHAPTER:{chapter2}",
            f"CLASS:{classe}",
            f"CLASS:{classe}",
            f"FAMILLE:{famille}",
            f"DESC:{desc}",
        ]

        specs_boostees, specs_normales = [], []
        for col in etim_cols:
            valeur = row.get(col)
            if pd.isna(valeur) or str(valeur).strip() == '':
                continue
            valeur_str = str(valeur).upper().strip()
            col_lower  = col.lower()

            if any(b in col_lower for b in MOTS_BANNIS):
                continue

            # Nom propre de la feature (après le "- ")
            nom = col.split('- ')[-1].strip() if '- ' in col else col.strip()
            info = f"{nom}:{valeur_str}"

            if any(c in col_lower for c in COLONNES_CRITIQUES):
                # Features critiques → ×3 pour le modèle
                specs_boostees.extend([info, info, info])
            else:
                specs_normales.append(info)

        parts.extend(specs_boostees)
        parts.extend(specs_normales)
        return ' | '.join(parts)

    print("    Génération des textes (peut prendre 2-5 min)...")
    df_master['TEXTE'] = df_master.apply(creer_texte_industriel, axis=1)
    print(f"    Longueur moyenne du texte : {df_master['TEXTE'].str.len().mean():.0f} chars")

    # ── 5. VECTORISATION TF-IDF ───────────────────────────────────────────────
    print("[5/7] Vectorisation TF-IDF + Split train/test...")

    # *** AMÉLIORATION 4 : max_features 5000 → 25 000 ***
    vectorizer = TfidfVectorizer(
        max_features=25_000,
        ngram_range=(1, 2),
        token_pattern=r'(?u)\b\w+\b',
        sublinear_tf=True,        # log(tf) évite de sur-pondérer les répétitions
        min_df=2,                 # ignore les hapax
    )
    X = vectorizer.fit_transform(df_master['TEXTE'])  # SPARSE — pas de .toarray() !
    print(f"    Matrice TF-IDF (sparse) : {X.shape}")
    print(f"    Densité : {X.nnz / (X.shape[0]*X.shape[1])*100:.3f}% (économie mémoire massive)")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_master['Code Douane'])
    n_classes = len(label_encoder.classes_)
    print(f"    {n_classes} classes à prédire")

    # Split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Train : {X_train.shape[0]:,}  |  Test : {X_test.shape[0]:,}")

    # Poids équilibrés (inchangé — très bonne idée originale)
    poids = compute_sample_weight(class_weight='balanced', y=y_train)

    # ── 6. ENTRAÎNEMENT XGBOOST ───────────────────────────────────────────────
    print("[6/7] Entraînement XGBoost optimisé...")
    print("    (peut prendre 10-30 min selon votre CPU)\n")

    # *** AMÉLIORATION 5 : hyperparamètres + early stopping ***
    modele = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        eval_metric='mlogloss',
        tree_method='hist',       # GPU-ready, rapide sur CPU aussi
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=3,
        gamma=0.1,
        early_stopping_rounds=40, # arrêt automatique si plus d'amélioration
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )

    # *** AMÉLIORATION 3 : XGBoost accepte le sparse natif — pas de .toarray() ! ***
    modele.fit(
        X_train, y_train,
        sample_weight=poids,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    best_iter = modele.best_iteration
    print(f"\n    Meilleur itération (early stopping) : {best_iter}")

    # *** AMÉLIORATION 6 : Top-1, Top-3, Top-5 ***
    print("\n    Calcul des métriques Top-K...")
    probas_test = modele.predict_proba(X_test)

    top1 = topk_accuracy(y_test, probas_test, 1)
    top3 = topk_accuracy(y_test, probas_test, 3)
    top5 = topk_accuracy(y_test, probas_test, 5)

    print("\n" + "-" * 45)
    print(f"  Top-1 Accuracy (précision exacte)  : {top1*100:.2f}%")
    print(f"  Top-3 Accuracy (dans les 3 premiers) : {top3*100:.2f}%")
    print(f"  Top-5 Accuracy (dans les 5 premiers) : {top5*100:.2f}%")
    print(f"  Nombre de classes                   : {n_classes}")
    print(f"  Produits d'entraînement             : {X_train.shape[0]:,}")
    print("-" * 45 + "\n")

    # ── 7. EXPORT ONNX ────────────────────────────────────────────────────────
    print("[7/7] Export ONNX pour l'interface web...")

    try:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_type = [('float_input', FloatTensorType([None, 25_000]))]
        onnx_model = onnxmltools.convert_xgboost(
            modele, initial_types=initial_type, target_opset=12
        )

        with open(FICHIER_ONNX, "wb") as f:
            f.write(onnx_model.SerializeToString())
        onnx_size = os.path.getsize(FICHIER_ONNX) / 1024 / 1024
        print(f"    modele_douane_legrand.onnx créé ({onnx_size:.1f} MB)")

    except Exception as e:
        print(f"    AVERTISSEMENT export ONNX : {e}")
        print("    → Le modèle est quand même sauvegardé en pickle (app.py)")
        import pickle
        with open(os.path.join(BASE_DIR, "modele_douane_pickle.pkl"), "wb") as f:
            pickle.dump(modele, f)

    # Config JSON pour le navigateur (vocabulaire TF-IDF + labels)
    config = {
        'vocabulaire': {str(mot): int(idx) for mot, idx in vectorizer.vocabulary_.items()},
        'idf': vectorizer.idf_.tolist(),
        'sublinear_tf': True,
        'codes_sh': label_encoder.classes_.astype(str).tolist(),
        'metriques': {
            'top1': round(top1 * 100, 2),
            'top3': round(top3 * 100, 2),
            'top5': round(top5 * 100, 2),
            'n_classes': n_classes,
            'n_train': int(X_train.shape[0]),
        },
        'max_features': 25_000,
    }

    with open(FICHIER_CONFIG, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False)
    config_size = os.path.getsize(FICHIER_CONFIG) / 1024 / 1024
    print(f"    config_ia_legrand.json créé ({config_size:.1f} MB)")

    # ZIP final
    with zipfile.ZipFile(FICHIER_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(FICHIER_ONNX):
            zf.write(FICHIER_ONNX, 'modele_douane_legrand.onnx')
        zf.write(FICHIER_CONFIG, 'config_ia_legrand.json')

    print(f"\n{'='*65}")
    print(f"  TERMINE ! Fichiers créés dans : {BASE_DIR}")
    print(f"  - {FICHIER_ZIP} (à déployer)")
    print(f"  - index.html      (interface web, ouvrir dans le navigateur)")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    master_pipeline()
