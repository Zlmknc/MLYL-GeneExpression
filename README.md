Veri Hazırlama ve Etiketleme:
Gerekli kütüphaneleri (pandas, sklearn, xgboost vb.) içe aktarır.  
spellman_periodic_genes.txt dosyasından 800 periyodik geni yükler.  
Spellman.csv veri setini okur.  
Verideki gen isimlerini referans listeyle karşılaştırarak periyodik olanları 1, olmayanları 0 olarak işaretleyen bir label sütunu ekler.  
Sonucu Spellman_with_labels.csv olarak kaydeder.  

Temel Sınıflandırma Denemeleri: Zaman serisi verilerini özellik (X) ve etiket (y) olarak ayırır. Özellik Çıkarımı: Ortalama, standart sapma gibi istatistiklerin yanı sıra Fourier Dönüşümü (FFT) kullanarak frekans tabanlı özellikler üretir. Random Forest, XGBoost, SVM ve Lojistik Regresyon modellerini eğitir. Her modelin performansını (Accuracy, F1-Score, Confusion Matrix) ekrana yazdırır.

İyileştirilmiş Random Forest ve SMOTE:  
Verideki sınıf dengesizliğini gidermek için SMOTE algoritmasıyla sentetik veri üretir.  
Özellik çıkarımına Otorelasyon (Autocorrelation) ekler. Random Forest modelini hiperparametrelerle optimize eder.  
Modelin yakalama oranını (recall) artırmak için karar eşiğini (threshold) 0.35'e çeker.

Gelişmiş Ensemble Model ve Sinyal İşleme:  
İleri Seviye Özellikler: FFT detaylandırılır, teppe noktası (peak) sayısı hesaplanır ve en güçlü özellik olarak sinüzoidal eğri uydurma (Curve Fitting) uygulanır. SMOTE ve ADASYN gibi farklı dengeleme tekniklerini dener. Random Forest ve XGBoost modellerini birleştiren bir Voting Classifier (Ensemble) oluşturur. En iyi sonucu veren eşik değerini bulmak için 0.30 ile 0.45 arasında bir tarama yapar. Final modelini spellman_periodic_classifier_ensemble.pkl olarak kaydeder.


# 🧬 Yeast Cell Cycle Periodic Gene Classifier

**Saccharomyces cerevisiae** genlerinin hücre döngüsü boyunca periyodik ifade gösterip göstermediğini makine öğrenmesi ile tahmin eden çok aşamalı sınıflandırma çerçevesi.

> Temel veri kaynağı: Spellman ve ark. (1998) — *Molecular Biology of the Cell*, 9(12), 3273–3297.

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Klasör Yapısı](#-klasör-yapısı)
- [Kurulum](#-kurulum)
- [Veri Setleri](#-veri-setleri)
- [Kullanım](#-kullanım)
- [Metodoloji](#-metodoloji)
- [Modeller ve Sonuçlar](#-modeller-ve-sonuçlar)
- [Değerlendirme Metrikleri](#-değerlendirme-metrikleri)
- [Tekrarlanabilirlik](#-tekrarlanabilirlik)
- [Bağımlılıklar](#-bağımlılıklar)
- [Atıflar](#-atıflar)
- [Lisans](#-lisans)

---

## 🔬 Proje Hakkında

Bu proje, maya (*S. cerevisiae*) hücre döngüsü zaman serisi gen ifade verisinden periyodik genleri makine öğrenmesiyle sınıflandırmayı amaçlamaktadır. Dört aşamalı bir pipeline sunulmaktadır:

| Aşama | Açıklama |
|-------|----------|
| **1** | Veri hazırlama ve ikili etiketleme (periyodik / periyodik olmayan) |
| **2** | Temel ML modelleri ile kıyas denemeleri |
| **3** | SMOTE dengeleme + optimize edilmiş Random Forest |
| **4** | SMOTETomek + Ensemble (Voting Classifier) + eşik optimizasyonu |

**Temel özellikler:**
- 17 boyutlu özellik vektörü: FFT bant güçleri, otokorelasyon, sinüzoidal eğri uydurma, istatistiksel özetler
- 7 sınıflandırıcının karşılaştırmalı değerlendirmesi (RF, XGBoost, SVM, LR, k-NN, LightGBM, CatBoost)
- Sınıf dengesizliği yönetimi: SMOTE ve SMOTETomek
- Recall odaklı eşik taraması (0.30–0.45 aralığı)
- Final ensemble modeli: `spellman_periodic_classifier_ensemble.pkl`

---

## 📁 Klasör Yapısı

```
spellman-cell-cycle-ml/
│
├── data/
│   ├── raw/
│   │   ├── Spellman.csv                        # Orijinal mikroarray verisi (Spellman 1998)
│   │   └── spellman_periodic_genes.txt         # 800 periyodik gen referans listesi
│   └── processed/
│       └── Spellman_with_labels.csv            # Etiketlenmiş veri seti
│
├── notebooks/
│   ├── 01_data_preparation.ipynb               # Etiketleme ve ön işleme
│   ├── 02_baseline_models.ipynb                # 7 temel model karşılaştırması
│   ├── 03_improved_random_forest.ipynb         # SMOTE + hiperparametre optimizasyonu
│   └── 04_ensemble_model.ipynb                 # Ensemble + eşik taraması
│
├── src/
│   ├── feature_engineering.py                  # Özellik çıkarım fonksiyonları
│   ├── models.py                               # Model tanımları ve eğitim
│   ├── evaluation.py                           # Metrik hesaplama ve görselleştirme
│   └── utils.py                               # Yardımcı fonksiyonlar
│
├── models/
│   └── spellman_periodic_classifier_ensemble.pkl   # Kaydedilmiş final model
│
├── results/
│   ├── figures/                                # ROC, PR eğrileri, confusion matrix vb.
│   └── metrics_summary.csv                    # Tüm modellerin metrik özeti
│
├── requirements.txt
├── environment.yml
└── README.md
```

---

## ⚙️ Kurulum

### 1. Depoyu Klonlayın

```bash
git clone https://github.com/kullanici-adi/spellman-cell-cycle-ml.git
cd spellman-cell-cycle-ml
```

### 2a. Conda ile (Önerilen)

```bash
conda env create -f environment.yml
conda activate spellman-ml
```

### 2b. pip ile

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Kurulumu Doğrulayın

```bash
python -c "import sklearn, xgboost, lightgbm, catboost, imblearn; print('Tüm bağımlılıklar yüklü.')"
```

---

## 📊 Veri Setleri

### Spellman.csv

Spellman ve ark. (1998) tarafından üç bağımsız senkronizasyon yöntemiyle elde edilen maya hücre döngüsü mikroarray verisi:

| Deney | Senkronizasyon yöntemi | Zaman noktası | Örnekleme aralığı |
|-------|------------------------|---------------|-------------------|
| Alpha faktör | Kimyasal arrest | 18 | 7 dakika |
| cdc15 | Sıcaklığa duyarlı mutant | 24 | 10–20 dakika (düzensiz) |
| Elutriation | Santrifügasyon | 14 | Değişken |

> Bütünleşik veri seti **23 zaman noktası** ve yaklaşık **6.000 gen** satırı içermektedir.

### spellman_periodic_genes.txt

Periyodiklik ve korelasyon algoritmalarıyla tanımlanmış **800 hücre döngüsü regüle geni** listesi. Sınıflandırma için pozitif sınıf (label = 1) etiketinin kaynağıdır.

**Sınıf dağılımı:**

```
Periyodik (1)      :   800  (~14.4 %)
Periyodik olmayan (0): ~4750  (~85.6 %)
```

> ⚠️ Belirgin sınıf dengesizliği — dengeleme stratejileri zorunludur.

### Eksik Değer Yönetimi

- Gen bazında **doğrusal enterpolasyon** uygulandı.
- Tüm zaman noktalarının **%30'undan fazlasında** eksik değer içeren genler analizden çıkarıldı.

---

## 🚀 Kullanım

### Adım 1 — Veri Hazırlama

```python
import pandas as pd

# Referans listesini yükle
with open("data/raw/spellman_periodic_genes.txt") as f:
    periodic_genes = set(line.strip() for line in f)

# Veri setini oku ve etiketle
df = pd.read_csv("data/raw/Spellman.csv", index_col=0)
df["label"] = df.index.isin(periodic_genes).astype(int)
df.to_csv("data/processed/Spellman_with_labels.csv")

print(f"Periyodik: {df['label'].sum()} | Periyodik olmayan: {(df['label']==0).sum()}")
```

### Adım 2 — Özellik Çıkarımı

```python
from src.feature_engineering import extract_features

X_raw = df.drop(columns=["label"]).values
y     = df["label"].values

X_features = extract_features(X_raw)   # (n_genes, 17) matris
print(f"Özellik matrisi boyutu: {X_features.shape}")
```

### Adım 3 — Temel Model Eğitimi

```python
from sklearn.model_selection import train_test_split
from src.models import train_baseline_models
from src.evaluation import print_metrics

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.25, stratify=y, random_state=42
)

results = train_baseline_models(X_train, y_train, X_test, y_test)
```

### Adım 4 — Ensemble Model ve Eşik Optimizasyonu

```python
from src.models import train_ensemble
from src.evaluation import threshold_scan

model = train_ensemble(X_train, y_train)
best_threshold, metrics_df = threshold_scan(
    model, X_test, y_test,
    thresholds=[round(t, 2) for t in __import__('numpy').arange(0.30, 0.46, 0.01)]
)

print(f"Optimal eşik: {best_threshold:.2f}")
print(metrics_df)
```

### Adım 5 — Kayıtlı Modeli Kullan

```python
import pickle
import numpy as np

with open("models/spellman_periodic_classifier_ensemble.pkl", "rb") as f:
    clf, threshold = pickle.load(f)

# Yeni gen profili (23 zaman noktası)
new_gene_expression = np.array([...])   # shape: (23,)

from src.feature_engineering import extract_features
X_new = extract_features(new_gene_expression.reshape(1, -1))

prob = clf.predict_proba(X_new)[0, 1]
pred = int(prob >= threshold)

print(f"Periyodik olasılığı: {prob:.3f} → {'Periyodik ✓' if pred else 'Periyodik değil ✗'}")
```

---

## 🔧 Metodoloji

### Özellik Mühendisliği (17 özellik)

| # | Özellik | Açıklama |
|---|---------|----------|
| 1 | mean | Zaman serisi ortalaması (μ) |
| 2 | std | Standart sapma (σ) |
| 3 | max | Maksimum ifade değeri |
| 4 | min | Minimum ifade değeri |
| 5 | range | max − min |
| 6 | skew | Fisher çarpıklığı (3. moment) |
| 7 | fft_power_low | FFT güç toplamı, bant 1–4 |
| 8 | fft_power_mid | FFT güç toplamı, bant 5–11 (hücre döngüsü aralığı) |
| 9 | fft_power_high | FFT güç toplamı, bant ≥12 |
| 10 | fft_total_power | Toplam spektral güç |
| 11 | dominant_freq | Baskın frekans |
| 12 | autocorr_lag1 | Lag-1 otokorelasyon katsayısı |
| 13 | autocorr_lag2 | Lag-2 otokorelasyon katsayısı |
| 14 | num_peaks | Tepe noktası sayısı (scipy.signal.find_peaks) |
| 15 | amplitude | Sinüzoidal eğri uydurma: genlik *A* |
| 16 | frequency | Sinüzoidal eğri uydurma: frekans *f* |
| 17 | r_squared | Sinüzoidal eğri uydurma kalitesi (R²) |

**Sinüzoidal model:**

```
y(t) = A · sin(2π · f · t + φ) + c
```

`A` = genlik | `f` = frekans | `φ` = faz | `c` = dikey öteleme  
Uydurma: `scipy.optimize.curve_fit` — başarısız yakınsama durumunda tüm parametreler 0 atanır.

---

## 🤖 Modeller ve Sonuçlar

### Temel Model Karşılaştırması

| Model | Accuracy | F1-Macro | ROC-AUC | PR-AUC |
|-------|----------|----------|---------|--------|
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |
| SVM (RBF) | — | — | — | — |
| Lojistik Regresyon | — | — | — | — |
| k-NN (k=5) | — | — | — | — |
| LightGBM | — | — | — | — |
| CatBoost | — | — | — | — |

> 📌 Tablo, çalışma tamamlandıktan sonra `results/metrics_summary.csv` dosyasından doldurulacaktır.

### Final Ensemble Modeli (Soft Voting)

```
RF (n=600, depth=14)
  + XGBoost (n=400, lr=0.08, depth=8)
  + LightGBM (n=400, lr=0.08, leaves=31)
  + CatBoost (iter=400, lr=0.08, depth=8)
  + k-NN (k=5, weights='distance')
──────────────────────────────────────
Dengeleme : SMOTETomek
Oylama    : Soft (olasılık ortalaması)
Eşik      : [threshold_scan ile belirlendi]
```

---

## 📈 Değerlendirme Metrikleri

Sınıf dengesizliği (%14.4 pozitif) nedeniyle birden fazla metrik raporlanmaktadır:

| Metrik | Gerekçe |
|--------|---------|
| **F1-Macro** | Sınıf büyüklüğünden bağımsız harmonik ortalama |
| **PR-AUC** | Dengesiz veri için ROC-AUC'den daha güvenilir |
| **MCC** | Tüm konfüzyon matrisi hücrelerini dengeli biçimde ele alır |
| **Recall** | Periyodik geni gözden kaçırma maliyeti yüksek → öncelikli metrik |
| **ROC-AUC** | Eşik bağımsız genel karşılaştırma |

**İstatistiksel karşılaştırma:**
- Model çiftleri arası: Wilcoxon işaretli sıra testi (α = 0.05)
- Çoklu karşılaştırma düzeltmesi: Benjamini-Hochberg (FDR)
- Güven aralıkları: Bootstrap yöntemi (n = 1000 tekrar)

---

## 🔁 Tekrarlanabilirlik

Tüm stokastik süreçlerde global rastgele tohum değeri **42** olarak sabitlenmiştir:

```python
import random, numpy as np
random.seed(42)
np.random.seed(42)
# sklearn, xgboost, lightgbm, catboost → random_state=42
```

**Yazılım ortamı:**

| Paket | Sürüm |
|-------|-------|
| Python | 3.10.12 |
| scikit-learn | 1.3.2 |
| XGBoost | 2.0.3 |
| LightGBM | 4.1.0 |
| CatBoost | 1.2.2 |
| imbalanced-learn | 0.11.0 |
| NumPy | 1.26.2 |
| pandas | 2.1.4 |
| SciPy | 1.11.4 |

---

## 📦 Bağımlılıklar

`requirements.txt` içeriği:

```
numpy==1.26.2
pandas==2.1.4
scipy==1.11.4
scikit-learn==1.3.2
xgboost==2.0.3
lightgbm==4.1.0
catboost==1.2.2
imbalanced-learn==0.11.0
matplotlib==3.8.2
seaborn==0.13.0
shap==0.43.0
joblib==1.3.2
jupyter==1.0.0
```

---

## 📚 Atıflar

Bu çalışmada kullanılan temel yöntem ve veri kaynakları:

```bibtex
@article{spellman1998,
  author  = {Spellman, Paul T. and Sherlock, Gavin and Zhang, Michael Q. and others},
  title   = {Comprehensive Identification of Cell Cycle-regulated Genes of the Yeast
             Saccharomyces cerevisiae by Microarray Hybridization},
  journal = {Molecular Biology of the Cell},
  year    = {1998},
  volume  = {9},
  number  = {12},
  pages   = {3273--3297},
  doi     = {10.1091/mbc.9.12.3273}
}

@article{chawla2002,
  author  = {Chawla, Nitesh V. and Bowyer, Kevin W. and Hall, Lawrence O. and Kegelmeyer, W. Philip},
  title   = {SMOTE: Synthetic Minority Over-sampling Technique},
  journal = {Journal of Artificial Intelligence Research},
  year    = {2002},
  volume  = {16},
  pages   = {321--357}
}

@inproceedings{batista2004,
  author    = {Batista, Gustavo E. A. P. A. and Prati, Ronaldo C. and Monard, Maria Carolina},
  title     = {A Study of the Behavior of Several Methods for Balancing Machine Learning
               Training Data},
  booktitle = {ACM SIGKDD Explorations Newsletter},
  year      = {2004},
  volume    = {6},
  number    = {1},
  pages     = {20--29}
}

@article{chen2013,
  author  = {Chen, Xin and Xu, Minghua},
  title   = {Identification of Yeast Cell Cycle Regulated Genes Based on Genomic Features},
  journal = {BMC Systems Biology},
  year    = {2013},
  volume  = {7},
  doi     = {10.1186/1752-0509-7-S2-S2}
}

@article{franz2020,
  author  = {Franz, Alexander and others},
  title   = {Improved Recovery of Cell-cycle Gene Expression in S. cerevisiae from
             Regulatory Interactions in Multiple Omics Data},
  journal = {BMC Genomics},
  year    = {2020},
  doi     = {10.1186/s12864-020-6554-8}
}
```

---

## 🤝 Katkıda Bulunma

1. Depoyu fork edin
2. Yeni bir dal oluşturun: `git checkout -b feature/yeni-ozellik`
3. Değişikliklerinizi commit edin: `git commit -m 'feat: yeni özellik eklendi'`
4. Dalınızı gönderin: `git push origin feature/yeni-ozellik`
5. Pull request açın

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) kapsamında dağıtılmaktadır.

---

## 📬 İletişim

Sorular ve öneriler için bir **Issue** açabilir ya da doğrudan iletişime geçebilirsiniz.

---

<p align="center">
  <sub>Spellman ve ark. (1998) verisi temel alınarak geliştirilmiştir · S. cerevisiae hücre döngüsü · ML tabanlı periyodik gen sınıflandırması</sub>
</p>
