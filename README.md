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
