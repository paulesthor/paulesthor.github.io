/**
 * webapp/app.js - Inference ONNX + TF-IDF in browser
 */

const state = {
  session: null,
  config: null,
  modelReady: false,
};

const DOM = {
  statModel: document.getElementById('statModel'),
  fileOnnx: document.getElementById('fileOnnx'),
  fileJson: document.getElementById('fileJson'),
  loadBtn: document.getElementById('loadModelBtn'),
  form: document.getElementById('predictForm'),
  submitBtn: document.getElementById('submitBtn'),
  featuresList: document.getElementById('techFeaturesList'),
  addFeatureBtn: document.getElementById('addFeatureBtn'),
  resultsPlaceholder: document.getElementById('resultsPlaceholder'),
  resultsContent: document.getElementById('resultsContent'),
  predictionCards: document.getElementById('predictionCards'),
  resultsQuery: document.getElementById('resultsQuery'),
  
  libelle: document.getElementById('libelle'),
  chapitre: document.getElementById('chapitre'),
  classeLeg: document.getElementById('classeLeg'),
  famille: document.getElementById('famille')
};

// ─── TF-IDF VECTORIZER JAVASCRIPT ────────────────────────────────────────────
class TfidfJS {
  constructor(config) {
    this.vocab = config.vocabulaire; // { "mot": idx }
    this.idf = config.idf; // [float, ...]
    this.numFeatures = config.max_features;
    this.sublinear = config.sublinear_tf;
  }

  tokenize(text) {
    text = text.toLowerCase();
    // (?u)\b\w+\b -> regex simple JS
    const words = text.match(/\b\w+\b/g) || [];
    const ngrams = [];
    
    // Unigrams
    for (let w of words) ngrams.push(w);
    // Bigrams
    for (let i = 0; i < words.length - 1; i++) {
      ngrams.push(words[i] + ' ' + words[i+1]);
    }
    return ngrams;
  }

  transform(text) {
    const tokens = this.tokenize(text);
    const tfMap = new Map(); // token -> count
    
    for (const t of tokens) {
      if (this.vocab.hasOwnProperty(t)) {
        const idx = this.vocab[t];
        tfMap.set(idx, (tfMap.get(idx) || 0) + 1);
      }
    }

    const vector = new Float32Array(this.numFeatures);
    let sumSq = 0.0;

    for (const [idx, count] of tfMap.entries()) {
      let tf = count;
      if (this.sublinear) {
        tf = 1.0 + Math.log(tf);
      }
      const weight = tf * this.idf[idx];
      vector[idx] = weight;
      sumSq += weight * weight;
    }

    // L2 Normalization
    if (sumSq > 0) {
      const norm = Math.sqrt(sumSq);
      for (let i = 0; i < this.numFeatures; i++) {
        if (vector[i] !== 0) vector[i] /= norm;
      }
    }

    return vector;
  }
}

// ─── MODEL LOADING ────────────────────────────────────────────────────────────

DOM.loadBtn.addEventListener('click', async () => {
  const onnxFile = DOM.fileOnnx.files[0];
  const jsonFile = DOM.fileJson.files[0];
  
  if (!onnxFile || !jsonFile) {
    alert("Veuillez sélectionner à la fois le modèle .onnx et la config .json");
    return;
  }
  
  try {
    DOM.statModel.querySelector('.stat-dot').className = 'stat-dot warning';
    DOM.statModel.querySelector('span:last-child').textContent = 'Chargement en cours...';
    
    // Lire le JSON
    const jsonText = await jsonFile.text();
    state.config = JSON.parse(jsonText);
    
    // Lire le ONNX et créer la session
    const arrayBuffer = await onnxFile.arrayBuffer();
    state.session = await ort.InferenceSession.create(arrayBuffer);
    
    // Init Vectorizer
    state.vectorizer = new TfidfJS(state.config);
    
    state.modelReady = true;
    DOM.submitBtn.disabled = false;
    
    DOM.statModel.querySelector('.stat-dot').className = 'stat-dot ready';
    DOM.statModel.querySelector('span:last-child').textContent = `Modèle prêt (${state.config.metriques.top1}% Top-1)`;
    
  } catch(e) {
    console.error(e);
    alert("Erreur de chargement du modèle: " + e.message);
    DOM.statModel.querySelector('.stat-dot').className = 'stat-dot error';
    DOM.statModel.querySelector('span:last-child').textContent = 'Erreur de chargement';
  }
});

// ─── DOM DYNAMIC ROWS ────────────────────────────────────────────────────────
function addFeatureRow() {
  const row = document.createElement('div');
  row.className = 'tech-feature-row';
  row.innerHTML = `
    <input type="text" class="field-input f-name" placeholder="Caractéristique (ex: couleur)">
    <input type="text" class="field-input f-val" placeholder="Valeur (ex: blanc)">
    <button type="button" class="btn-remove-feature" title="Supprimer">✕</button>
  `;
  row.querySelector('.btn-remove-feature').onclick = () => row.remove();
  DOM.featuresList.appendChild(row);
}
DOM.addFeatureBtn.addEventListener('click', addFeatureRow);
addFeatureRow(); // One row by default

// ─── PREDICTION ──────────────────────────────────────────────────────────────

const CRITIQUES = ['material', 'current', 'voltage', 'ampere', 'pole', 'mounting', 'type', 'power', 'protection', 'degree', 'rated', 'breaking', 'sensitivity', 'frequency'];
const BANNIS = ['colour', 'color', 'ral', 'packaging', 'storage', 'date', 'status', 'finishing', 'finish', 'packing', 'gross', 'net', 'weight', 'carton', 'pallet', 'ean', 'gtin'];

function buildText(libelle, chapter, classeLeg, famille, features) {
  let parts = [
    `CHAPTER:${chapter}`, `CHAPTER:${chapter}`,
    `CLASS:${classeLeg}`, `CLASS:${classeLeg}`,
    `FAMILLE:${famille}`,
    `DESC:${libelle.toUpperCase()}`
  ];
  
  let boost = [];
  let norm = [];
  
  features.forEach(f => {
    let name = f.name.toLowerCase();
    if (BANNIS.some(b => name.includes(b))) return;
    
    let info = `${f.name.toUpperCase()}:${f.val.toUpperCase()}`;
    if (CRITIQUES.some(c => name.includes(c))) {
      boost.push(info, info, info);
    } else {
      norm.push(info);
    }
  });
  
  return parts.concat(boost, norm).join(' | ');
}

DOM.form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!state.modelReady) return;
  
  const libelle = DOM.libelle.value.trim();
  const chapter = DOM.chapitre.value.trim().substring(0,2) || '';
  const classeLeg = (DOM.classeLeg.value || '').toUpperCase();
  const famille = (DOM.famille.value || '').toUpperCase();
  
  const featureInputs = document.querySelectorAll('.tech-feature-row');
  const features = [];
  featureInputs.forEach(row => {
    const name = row.querySelector('.f-name').value.trim();
    const val = row.querySelector('.f-val').value.trim();
    if (name && val) features.push({name, val});
  });
  
  const texte = buildText(libelle, chapter, classeLeg, famille, features);
  console.log("Texte généré :", texte);
  
  // Vectorize
  const vec = state.vectorizer.transform(texte);
  
  // Inference
  // ONNX JS usually expects tensors. 
  // 'float_input' is the input name expected in ONNX (check conversion script)
  const tensor = new ort.Tensor('float32', vec, [1, vec.length]);
  
  try {
    const feeds = { float_input: tensor };
    const results = await state.session.run(feeds);
    
    // L'output ONNX pour XGBoost multi-classe proba peut varier. 
    // Souvent 'probabilities' ou le second output contient une map ou un array de probas.
    // Analysons les clés
    const outputNames = state.session.outputNames; 
    let probas = [];
    if (results['probabilities']) {
      // Map(int64, float32) or tensor
      const probaTensor = results['probabilities'];
      // Si la sortie est de type map sequence (rare JS), c'est complexe.
      // S'il s'agit d'un tenseur de shape [1, N_classes]:
      if (probaTensor.data && probaTensor.data.length === state.config.codes_sh.length) {
        probas = Array.from(probaTensor.data);
      } else {
        console.warn("Format proba inattendu:", probaTensor);
        // Si c'est un tableau de Dicts (onnx-ml default multi-class output)
        if (Array.isArray(probaTensor)) {
            // usually [{0L: 0.1, 1L: 0.9}] array map
            // En js, ort web ne retourne pas souvent array map, mais un tenseur [1, n_classes]
            // check structure
        }
      }
    } else {
      // Fallback if 'probabilities' isn't there, check output names
      for (let n of outputNames) {
        if (results[n].data && results[n].data.length === state.config.codes_sh.length) {
          probas = Array.from(results[n].data);
          break;
        }
      }
    }
    
    if (!probas.length) {
       throw new Error("Impossible d'extraire les probabilités de la sortie ONNX");
    }
    
    // Trier et afficher top 5
    const classScores = probas.map((p, i) => ({code: state.config.codes_sh[i], proba: p}));
    classScores.sort((a,b) => b.proba - a.proba);
    const top5 = classScores.slice(0,5);
    
    renderResults(libelle, top5);
    
  } catch(err) {
    console.error(err);
    alert("Erreur lors de la prédiction : " + err.message);
  }
});

function renderResults(query, top5) {
  DOM.resultsPlaceholder.hidden = true;
  DOM.resultsContent.hidden = false;
  DOM.resultsQuery.textContent = `"${query}"`;
  
  DOM.predictionCards.innerHTML = '';
  top5.forEach((pred, idx) => {
    const p = (pred.proba * 100).toFixed(1);
    const rank = idx + 1;
    const card = document.createElement('div');
    card.className = `pred-card ${rank === 1 ? 'rank-1' : ''}`;
    
    let confClass = p >= 70 ? 'high' : p >= 40 ? 'medium' : 'low';
    let codeStr = pred.code;
    let formatCode = codeStr.length >= 8 ? `${codeStr.substring(0,4)} ${codeStr.substring(4,6)} ${codeStr.substring(6,8)} ${codeStr.substring(8)}` : codeStr;
    
    card.innerHTML = `
      <div class="pred-rank">${rank}</div>
      <div class="pred-body">
        <div class="pred-code">${formatCode}</div>
        <div class="pred-chapters">
          <span class="chapter-tag" style="font-size:0.8rem; background:rgba(255,255,255,0.05); padding: 2px 6px; border-radius:4px;">Chapitre ${codeStr.substring(0,2)}</span>
        </div>
      </div>
      <div class="pred-confidence">
        <div class="confidence-value ${confClass}">${p}%</div>
        <div class="confidence-bar-track">
          <div class="confidence-bar-fill fill-${confClass}" style="width: ${p}%"></div>
        </div>
      </div>
    `;
    DOM.predictionCards.appendChild(card);
  });
}
