// Génère presentation/LoRA_PEFT_RAHANETRA.pptx à partir de results/*.json
// Usage : NODE_PATH=<dossier node_modules> node presentation/build_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.APPLY_THEME);

const ROOT = path.resolve(__dirname, "..");
const R = (n) => JSON.parse(fs.readFileSync(path.join(ROOT, "results", `${n}.json`), "utf8"));
const runs = { head: R("head"), lora_r4: R("lora_r4"), lora_r8: R("lora_r8"), lora_r16: R("lora_r16"),
  lora_scratch_r8: R("lora_scratch_r8"), full: R("full") };
const LBL = { head: "Tête seule", lora_r4: "LoRA r=4", lora_r8: "LoRA r=8", lora_r16: "LoRA r=16",
  lora_scratch_r8: "LoRA maison r=8", full: "Fine-tuning complet" };
const KEYS = Object.keys(runs);
const acc = (k) => +(100 * runs[k].metrics.accuracy).toFixed(2);
const fr = (x, d = 1) => x.toLocaleString("fr-FR", { minimumFractionDigits: d, maximumFractionDigits: d });
const P = (k) => runs[k].profiling;
const M = runs.full.machine;

const THEME = {
  name: "LoRA Teal",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1D2B36", lt1: "FFFFFF", dk2: "0E3B43", lt2: "EEF4F3",
    accent1: "0F766E", accent2: "E39B2D", accent3: "5FA8A0", accent4: "C2553A",
    accent5: "8FA3AD", accent6: "2C4A52", hlink: "0F766E", folHlink: "2C4A52",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = "RAHANETRA Fanasina Jason";
pres.title = "LoRA — Parameter-Efficient Fine-Tuning";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const FOOT = "RAHANETRA F. J. · M1 IA & Big Data · Python avancé";

pres.defineSlideMaster({
  title: "TITRE_SOMBRE",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.1, w: 11.7, h: 1.6, fontSize: 44, bold: true,
      color: C.background1, fontFace: THEME.headFontFace, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.9, w: 11.7, h: 1.0, fontSize: 20,
      color: C.accent3, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENU",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.4, w: 12.1, h: 0.9, fontSize: 34, bold: true,
      color: C.text2, fontFace: THEME.headFontFace, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: FOOT, options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});

const content = (title, section) => {
  const s = pres.addSlide({ masterName: "CONTENU", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
};
const card = (s, name, x, y, w, h, fill = C.background2) =>
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.12, objectName: name });
const txt = (s, text, opts) => s.addText(text, { isTextBox: true, margin: 0, valign: "top", ...opts });
const FORMULA_IN_PER_PX = 0.0043;
const formula = (s, name, x, yCenter, scale = 1) => {
  const file = path.join(ROOT, "figures", "formulas", `${name}.png`);
  const { width, height } = pngSize(file);
  const h = height * FORMULA_IN_PER_PX * scale;
  s.addImage({ path: file, x, y: yCenter - h / 2, h, w: width * FORMULA_IN_PER_PX * scale, objectName: `formule-${name}` });
};
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { width: b.readUInt32BE(16), height: b.readUInt32BE(20) };
}
const AXIS = { catAxisLabelColor: "1D2B36", valAxisLabelColor: "5B6B73", catAxisLabelFontFace: "+mn-lt",
  valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt", catAxisLabelFontSize: 11,
  valAxisLabelFontSize: 10, dataLabelFontSize: 11, dataLabelColor: "1D2B36", titleFontSize: 14, titleColor: "1D2B36",
  valGridLine: { color: "DDE5E4", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false };
const runColor = (k) => (k === "full" ? HEX.accent4 : k === "head" ? HEX.accent5 : k === "lora_scratch_r8" ? HEX.accent2 : HEX.accent1);

// 1. Titre
pres.addSection({ title: "Introduction" });
{
  const s = pres.addSlide({ masterName: "TITRE_SOMBRE", sectionTitle: "Introduction" });
  s.addText("Adapter un modèle de langage en n'entraînant que 1 % de ses poids", { placeholder: "title" });
  s.addText("La méthode PEFT–LoRA appliquée à DistilBERT · classification de sentiments SST-2", { placeholder: "body" });
  txt(s, [
    { text: "RAHANETRA Fanasina Jason", options: { bold: true, breakLine: true } },
    { text: "Master 1 Intelligence Artificielle et Big Data · Python avancé", options: { breakLine: true } },
    { text: "Enseignant : RATIARISON Tsinto Aina · Octobre 2026" },
  ], { x: 0.8, y: 5.6, w: 9, h: 1.1, fontSize: 14, color: C.background1, objectName: "auteur" });
  s.addNotes("Présentation du projet final : LoRA, une méthode d'adaptation efficace en paramètres.");
}

// 2. Problème
{
  const s = content("Le fine-tuning complet coûte trop cher en mémoire et en stockage", "Introduction");
  const stats = [
    ["67 M", "paramètres dans DistilBERT, tous mis à jour"],
    ["16 octets", "par paramètre entraînable : poids, gradient, 2 moments Adam"],
    [`${fr(P("full").checkpoint_size_mb / 1, 0)} Mo`, "à sauvegarder pour chaque nouvelle tâche"],
  ];
  stats.forEach(([big, small], i) => {
    const x = 0.6 + i * 4.1;
    card(s, `stat-${i}`, x, 1.7, 3.8, 3.0);
    txt(s, big, { x: x + 0.3, y: 1.95, w: 3.2, h: 1.1, fontSize: 48, bold: true, color: C.accent4, fontFace: THEME.headFontFace, valign: "middle" });
    txt(s, small, { x: x + 0.3, y: 3.2, w: 3.2, h: 1.3, fontSize: 17, color: C.text1 });
  });
  txt(s, [
    { text: "Question de recherche : ", options: { bold: true, color: C.accent1 } },
    { text: "peut-on adapter un modèle pré-entraîné en n'entraînant qu'une petite fraction de ses paramètres, sans perdre en précision, sur une carte de 4 Go ?" },
  ], { x: 0.6, y: 5.2, w: 12.1, h: 1.3, fontSize: 20, color: C.text1 });
  s.addNotes("Pour un LLM de 7 milliards de paramètres, le fine-tuning complet dépasse 100 Go de mémoire GPU.");
}

// 3. État de l'art
{
  const s = content("Trois familles de méthodes PEFT, LoRA ne coûte rien à l'inférence", "Introduction");
  const fam = [
    ["Adapters", "Houlsby 2019", "Petits réseaux insérés dans chaque bloc", "Latence ajoutée à l'inférence"],
    ["Prompts appris", "Li & Liang 2021 · Lester 2021", "Vecteurs appris ajoutés à l'entrée", "Instables, consomment la séquence"],
    ["LoRA", "Hu et al. 2022", "Mise à jour de rang faible ΔW = BA", "Fusionnable : zéro surcoût"],
  ];
  fam.forEach(([name, ref, how, lim], i) => {
    const x = 0.6 + i * 4.1;
    const hi = i === 2;
    card(s, `famille-${i}`, x, 1.6, 3.8, 3.6, hi ? C.accent1 : C.background2);
    const col = hi ? C.background1 : C.text1;
    txt(s, name, { x: x + 0.3, y: 1.85, w: 3.2, h: 0.6, fontSize: 24, bold: true, color: hi ? C.background1 : C.text2, fontFace: THEME.headFontFace });
    txt(s, ref, { x: x + 0.3, y: 2.45, w: 3.2, h: 0.4, fontSize: 12, italic: true, color: hi ? C.background2 : C.accent5 });
    txt(s, how, { x: x + 0.3, y: 3.05, w: 3.2, h: 0.9, fontSize: 16, color: col });
    txt(s, lim, { x: x + 0.3, y: 4.1, w: 3.2, h: 0.8, fontSize: 16, bold: true, color: hi ? C.accent2 : C.accent4 });
  });
  txt(s, "Variantes récentes : QLoRA (base 4 bits), AdaLoRA (rang adaptatif), DoRA (amplitude + direction).",
    { x: 0.6, y: 5.6, w: 12.1, h: 0.5, fontSize: 15, color: C.text1 });
  s.addNotes("LoRA s'appuie sur la faible dimension intrinsèque des modèles pré-entraînés (Aghajanyan 2021).");
}

// 4. Principe LoRA (schéma)
pres.addSection({ title: "Méthodologie" });
{
  const s = content("LoRA gèle W₀ et n'apprend que deux petites matrices A et B", "Méthodologie");
  // schéma : x -> [W0 gelé] + [A -> B] -> h
  const y0 = 1.9;
  txt(s, "x", { x: 0.7, y: y0 + 1.15, w: 0.6, h: 0.6, fontSize: 28, italic: true, color: C.text1, fontFace: THEME.headFontFace, valign: "middle", align: "center" });
  s.addShape(pres.shapes.RECTANGLE, { x: 1.8, y: y0, w: 2.6, h: 2.9, fill: { color: "C9D3D6" }, line: { color: "C9D3D6" }, objectName: "W0" });
  txt(s, [{ text: "W₀", options: { bold: true, fontSize: 30, breakLine: true } }, { text: "768 × 768 · gelé", options: { fontSize: 13 } }],
    { x: 1.8, y: y0 + 0.9, w: 2.6, h: 1.1, color: C.text1, align: "center", valign: "middle" });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.0, y: y0, w: 2.6, h: 0.55, fill: { color: C.accent2 }, line: { color: C.accent2 }, objectName: "A" });
  txt(s, "A  (r × 768)", { x: 5.0, y: y0, w: 2.6, h: 0.55, fontSize: 14, bold: true, color: C.text1, align: "center", valign: "middle" });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.0, y: y0 + 0.75, w: 0.55, h: 2.15, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "B" });
  txt(s, "B (768 × r) = 0 au départ", { x: 5.75, y: y0 + 1.3, w: 1.9, h: 1.1, fontSize: 13, bold: true, color: C.accent1 });
  txt(s, "+", { x: 8.0, y: y0 + 1.1, w: 0.6, h: 0.7, fontSize: 36, bold: true, color: C.text1, align: "center", valign: "middle" });
  txt(s, "h", { x: 8.9, y: y0 + 1.15, w: 0.6, h: 0.6, fontSize: 28, italic: true, color: C.text1, fontFace: THEME.headFontFace, valign: "middle", align: "center" });
  formula(s, "lora", 0.7, 5.6, 1.15);
  txt(s, [
    { text: "Entraîné : ", options: { bold: true, color: C.accent1 } },
    { text: "A et B sur Wq et Wv des 6 blocs, plus la tête de classification", options: { breakLine: true } },
    { text: "Gelé : ", options: { bold: true, color: C.accent4 } },
    { text: "les 66 M autres paramètres" },
  ], { x: 9.8, y: 1.9, w: 2.9, h: 2.9, fontSize: 15, color: C.text1, paraSpaceAfter: 8 });
  s.addNotes("Le produit BA n'est jamais matérialisé pendant l'entraînement : on calcule (x Aᵀ) Bᵀ.");
}

// 5. Mathématiques
{
  const s = content("Moins de 2 % des paramètres d'une matrice, et fusion sans latence", "Méthodologie");
  const rows = [
    ["Initialisation", "init"],
    ["Ratio de paramètres", "params"],
    ["Gradients (s = α/r)", "grad"],
    ["Mémoire AdamW (fp32)", "memory"],
    ["Fusion pour l'inférence", "merge"],
  ];
  rows.forEach(([lab, f], i) => {
    const y = 1.55 + i * 1.03;
    card(s, `ligne-${i}`, 0.6, y, 12.1, 0.88, i % 2 ? C.background1 : C.background2);
    txt(s, lab, { x: 0.9, y, w: 3.4, h: 0.88, fontSize: 16, bold: true, color: C.text2, valign: "middle" });
    formula(s, f, 4.5, y + 0.44);
  });
  s.addNotes("Sur DistilBERT : 18 432·r paramètres LoRA + 592 130 pour la tête. Pour r = 8 : 739 586, soit 1,10 %.");
}

// 6. Protocole & machine
{
  const s = content("Protocole : 6 configurations, mêmes données, même machine", "Méthodologie");
  card(s, "protocole", 0.6, 1.6, 6.0, 4.9);
  txt(s, "Protocole", { x: 0.9, y: 1.8, w: 5.4, h: 0.5, fontSize: 20, bold: true, color: C.accent1, fontFace: THEME.headFontFace });
  txt(s, [
    { text: "DistilBERT-base-uncased · SST-2", options: { bullet: true, breakLine: true } },
    { text: "20 000 phrases d'entraînement, 872 de validation", options: { bullet: true, breakLine: true } },
    { text: "2 époques, lots de 32, 128 jetons max, AdamW", options: { bullet: true, breakLine: true } },
    { text: "LoRA sur Wq, Wv · α = 16 · r ∈ {4, 8, 16}", options: { bullet: true, breakLine: true } },
    { text: "Comparés : tête seule, fine-tuning complet, LoRA PEFT, LoRA maison", options: { bullet: true } },
  ], { x: 0.9, y: 2.5, w: 5.5, h: 3.8, fontSize: 17, color: C.text1, paraSpaceAfter: 14 });
  const rowsM = [
    ["GPU", `${M.gpu} · 4 Go`],
    ["Calcul CUDA", `CUDA ${M.cuda} · capacité ${M.compute_capability}`],
    ["CPU", M.cpu.replace(/\(R\)|\(TM\)|CPU /g, "").replace(/\s+/g, " ")],
    ["RAM", `${M.ram_gb.replace(".", ",")} Go`],
    ["Système", M.os],
    ["Logiciels", `Python ${M.python} · PyTorch ${M.torch.split("+")[0]}`],
  ];
  txt(s, "Configuration de l'ordinateur", { x: 7.0, y: 1.8, w: 5.7, h: 0.5, fontSize: 20, bold: true, color: C.accent1, fontFace: THEME.headFontFace });
  s.addTable(rowsM.map(([k, v]) => [
    { text: k, options: { bold: true, color: C.text2 } }, { text: v, options: { color: C.text1 } }]),
  { x: 7.0, y: 2.45, w: 5.7, colW: [1.6, 4.1], fontSize: 14, border: { type: "solid", pt: 0.5, color: "DDE5E4" }, rowH: 0.62, valign: "middle", objectName: "table-machine" });
  s.addNotes("Toutes les durées sont mesurées avec synchronisation CUDA ; mémoire = torch.cuda.max_memory_allocated.");
}

// 7. Résultats précision
pres.addSection({ title: "Résultats" });
{
  const best = Math.max(acc("lora_r4"), acc("lora_r8"), acc("lora_r16"));
  const s = content(`LoRA atteint ${fr(best, 1)} % de précision avec ~1 % des paramètres`, "Résultats");
  s.addChart(pres.charts.BAR, [{ name: "Accuracy (%)", labels: KEYS.map((k) => LBL[k]), values: KEYS.map(acc) }], {
    x: 0.6, y: 1.5, w: 7.6, h: 5.2, barDir: "bar", chartColors: KEYS.map(runColor), ...AXIS,
    showTitle: true, title: "Accuracy sur la validation SST-2 (%)", showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.00", valAxisMinVal: 80, valAxisMaxVal: 92, catAxisOrientation: "maxMin", objectName: "graphe-accuracy",
  });
  const tr = (k) => runs[k].params.trainable;
  const rows = [["Méthode", "Param. entraînables", "F1 (%)"]].concat(KEYS.map((k) => [LBL[k], tr(k).toLocaleString("fr-FR"), fr(100 * runs[k].metrics.f1, 2)]));
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0, color: i === 0 ? C.background1 : C.text1,
    fill: { color: i === 0 ? C.accent1 : i % 2 ? C.background2 : C.background1 }, align: j ? "right" : "left" } }))),
  { x: 8.5, y: 1.7, w: 4.2, colW: [1.75, 1.45, 1.0], fontSize: 11, rowH: 0.45, border: { type: "none" }, valign: "middle", objectName: "table-resultats" });
  txt(s, `Le fine-tuning complet entraîne ${Math.floor(tr("full") / tr("lora_r8"))} × plus de paramètres que LoRA r=8.`,
    { x: 8.5, y: 5.2, w: 4.2, h: 1.0, fontSize: 14, italic: true, color: C.accent1 });
  s.addNotes("La LoRA maison et la LoRA PEFT donnent des résultats proches : l'implémentation est validée.");
}

// 8. Profilage
{
  const s = content("Profilage : LoRA réduit la mémoire GPU et le stockage", "Résultats");
  const common = { ...AXIS, chartColors: KEYS.map(runColor), showValue: true, dataLabelPosition: "outEnd", showTitle: true };
  s.addChart(pres.charts.BAR, [{ name: "min", labels: KEYS.map((k) => LBL[k]), values: KEYS.map((k) => +(P(k).train_time_s / 60).toFixed(1)) }],
    { ...common, x: 0.5, y: 1.5, w: 4.1, h: 4.6, title: "Durée d'entraînement (min)", dataLabelFormatCode: "0.0", catAxisLabelRotate: -40, objectName: "graphe-temps" });
  s.addChart(pres.charts.BAR, [{ name: "Mo", labels: KEYS.map((k) => LBL[k]), values: KEYS.map((k) => Math.round(P(k).gpu_peak_train_mb)) }],
    { ...AXIS, chartColors: KEYS.map(runColor), showValue: true, dataLabelPosition: "outEnd", showTitle: true,
      x: 4.65, y: 1.5, w: 4.1, h: 4.6, title: "Pic mémoire GPU (Mo)", dataLabelFormatCode: "0", catAxisLabelRotate: -40, objectName: "graphe-memoire" });
  s.addChart(pres.charts.BAR, [{ name: "Mo", labels: KEYS.map((k) => LBL[k]), values: KEYS.map((k) => +P(k).checkpoint_size_mb.toFixed(1)) }],
    { ...AXIS, chartColors: KEYS.map(runColor), showValue: true, dataLabelPosition: "outEnd", showTitle: true,
      x: 8.8, y: 1.5, w: 4.1, h: 4.6, title: "Taille sauvegardée (Mo)", dataLabelFormatCode: "0.0", catAxisLabelRotate: -40, objectName: "graphe-disque" });
  const memGain = (1 - P("lora_r8").gpu_peak_train_mb / P("full").gpu_peak_train_mb) * 100;
  const timeGain = (1 - P("lora_r8").train_time_s / P("full").train_time_s) * 100;
  txt(s, `LoRA r=8 face au fine-tuning complet : −${fr(memGain, 0)} % de mémoire GPU, −${fr(timeGain, 0)} % de temps, un checkpoint ${fr(P("full").checkpoint_size_mb / P("lora_r8").checkpoint_size_mb, 0)} × plus léger.`,
    { x: 0.6, y: 6.25, w: 12.1, h: 0.6, fontSize: 15, bold: true, color: C.accent1 });
  s.addNotes("Mesures réelles sur la GTX 1050. Le checkpoint LoRA contient l'adaptateur et la tête de classification.");
}

// 9. Application
pres.addSection({ title: "Conclusion" });
{
  const s = content("Application fonctionnelle : analyse de sentiment en direct", "Conclusion");
  const shot = path.join(ROOT, "figures", "app_screenshot.png");
  let shotH = 5.0;
  if (fs.existsSync(shot)) {
    const { width, height } = pngSize(shot);
    const w = 8.3;
    shotH = (w * height) / width;
    s.addImage({ path: shot, x: 0.6, y: 1.55, w, h: (w * height) / width, objectName: "capture-app",
      shadow: { type: "outer", color: "1D2B36", opacity: 0.25, blur: 8, offset: 3, angle: 90 } });
  }
  card(s, "app-info", 9.2, 1.55, 3.5, Math.max(shotH, 4.2));
  txt(s, [
    { text: "Gradio + PEFT", options: { bold: true, fontSize: 20, color: C.accent1, breakLine: true } },
    { text: "Choix de l'adaptateur r = 4, 8 ou 16", options: { bullet: true, breakLine: true } },
    { text: "Fusion W₀ + (α/r)BA au chargement", options: { bullet: true, breakLine: true } },
    { text: "Probabilités et latence affichées", options: { bullet: true, breakLine: true } },
    { text: "Métriques du modèle affichées", options: { bullet: true, breakLine: true } },
    { text: "Lancement : python app.py", options: { breakLine: true, fontFace: "Courier New", fontSize: 13 } },
  ], { x: 9.45, y: 1.8, w: 3.05, h: 4.2, fontSize: 15, color: C.text1, paraSpaceAfter: 8 });
  s.addNotes("Démonstration en direct de l'application pendant la soutenance.");
}

// 10. Conclusion
{
  const s = pres.addSlide({ masterName: "TITRE_SOMBRE", sectionTitle: "Conclusion" });
  s.addText("LoRA : la précision du fine-tuning complet, pour une fraction du coût", { placeholder: "title" });
  s.addText(`${fr(acc("lora_r4"), 1)} % (LoRA r=4) contre ${fr(acc("full"), 1)} % : −44 % de mémoire GPU, −32 % de temps, checkpoint 90 × plus léger`, { placeholder: "body" });
  txt(s, [
    { text: "Perspectives : ", options: { bold: true, color: C.accent2 } },
    { text: "QLoRA pour un LLM sur 4 Go · LoRA sur toutes les couches · rang adaptatif (AdaLoRA) · tâches génératives" },
  ], { x: 0.8, y: 5.4, w: 11.7, h: 1.0, fontSize: 18, color: C.background1 });
  s.addNotes("Merci. Questions ?");
}

(async () => {
  const out = path.join(__dirname, "LoRA_PEFT_RAHANETRA.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("écrit :", out);
})();
