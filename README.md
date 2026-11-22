# Klasik görüntü işleme ile mikroskobik görüntülerde hücre sayımı

İbrahim Alpsoy, 254329040
Ders: Görüntü İşleme (Doç. Dr. Gökalp Tulum)

Bu projede Cell Tracking Challenge veri setindeki Fluo-N2DH-GOWT1 mikroskop görüntülerinde hücreleri sadece klasik görüntü işleme yöntemleriyle (eşikleme, filtreleme, morfoloji vb.) ayırıp saydım. Farklı yoğunluk dönüşümlerini, filtreleri, eşikleme yöntemlerini ve morfolojik işlemleri geliştirme karelerinde karşılaştırdım, 3 aday pipeline'ı doğrulama karelerinde deneyip birini seçtim ve seçtiğim yöntemi test karelerinde baseline ile karşılaştırdım. Derin öğrenme kullanılmadı.

## Veri seti

Resmî indirme linki (training arşivi, 53 MB):
https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip

Veri seti sayfası: https://celltrackingchallenge.net/2d-datasets/

Veri seti repoda yok, çünkü orijinal görüntülerin ve maskelerin paylaşılmasına izin verilmiyor. İndirdiğin zip dosyasını `data` klasörüne koyman yeterli, notebook zip'i kendisi açıyor. Elle açmak istersen klasör yapısı `data/README.md` içinde yazıyor.

## Seed

Seed: 29040 + 2026 = **31066** (öğrenci numaramın son 5 hanesi + 2026)

Kare seçimi bu seed ile `src/secim.py` içinde yapılıyor:
- Sequence 01: 10 geliştirme ve 5 doğrulama karesi. İki grup arasında en az 5 kare boşluk var.
- Sequence 02: 3 test bloğu, her biri ardışık 5 kare. Bloklar arasında en az 5 kare boşluk var.
- Koşul sağlanmazsa aynı rastgele üreteçle tekrar çekiliyor.

Seçilen kareler `selected_frames.csv` dosyasında. Seçimi algoritmaya başlamadan önce yapıp ilk commit ile kaydettim.

## Gereksinimler

Python 3.13 ile çalıştırdım. Kütüphaneler ve sürümler `requirements.txt` içinde: numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, scikit-image 0.26.0, matplotlib 3.11.2, opencv-python-headless 5.0.0.

```bash
git clone https://github.com/ibrahimalpsoy/goruntu-isleme-hucre-sayimi.git
cd goruntu-isleme-hucre-sayimi
pip install -r requirements.txt
```

## Çalıştırma

**Jupyter:** `Fluo-N2DH-GOWT1.zip` dosyasını `data` klasörüne koy ve `main.ipynb` dosyasını baştan sona çalıştır.

**Google Colab:** `main.ipynb` dosyasını aç ve ilk hücreyi çalıştır, repoyu klonlayıp paketleri kuruyor. Sonra zip dosyasını Colab'da `goruntu-isleme-hucre-sayimi/data/` klasörüne yükle ve kalan hücreleri sırayla çalıştır.

`main.ipynb` içindeki sıra şöyle:
1. Veri inceleme (boyut, veri tipi, yoğunluklar, histogramlar, hücre boyutu ve parlaklığı)
2. Kare seçimi (seed 31066)
3. Geliştirme ve doğrulama karelerinin referans hücre sayıları (`TRA` etiketlerindeki farklı hücre kimlikleri)
4. Baseline (gri görüntü + Otsu + bağlı bileşen analizi)
5. Parametre taramaları ve 22 kontrollü deney (E01-E22)
6. 3 aday pipeline'ın doğrulama karşılaştırması
7. Nihai pipeline'ın `final_pipeline.json` dosyasına kaydedilmesi
8. Test tahminlerinin referanslara bakılmadan `test_predictions_locked.csv` dosyasına yazılması
9. Test değerlendirmesi, zamansal analiz ve hata analizi

Kare seçimi sadece `src/secim.py` ile de çalıştırılabilir: `python -m src.secim`

## Nihai pipeline

Doğrulamada en düşük MAE değerini Pipeline A ve Pipeline B aynı aldı (1.4). Eşitlikte RMSE daha düşük olduğu için Pipeline A seçildi:

orijinal gri görüntü, Gaussian 5x5, sabit eşik T=3, açma (disk r=3), delik doldurma, 400 pikselden küçük nesneleri silme, bağlı bileşen etiketleme ve sayım

Bütün parametreler `final_pipeline.json` dosyasında.

Kısaca sonuçlar (15 test karesi):

- Baseline: MAE 5.6, RMSE 6.12, Bias -1.2, tam doğru sayım %0, ortalama süre 0.020 sn
- Nihai pipeline: MAE 3.0, RMSE 3.28, Bias -3.0, tam doğru sayım %6.7, ortalama süre 0.046 sn

## Dosyalar

- `main.ipynb`: bütün çalışma
- `src/`: veri okuma (`veri.py`), kare seçimi (`secim.py`), görüntü işleme adımları (`islemler.py`), deney tanımları ve metrikler (`deney.py`), ayarlar (`ayarlar.py`)
- `selected_frames.csv`: seçilen kareler, grupları, test blokları ve seed
- `experiments.csv`: baseline ve 22 kontrollü deney (yoğunluk, filtre, segmentasyon, morfoloji, parametreler, MAE, RMSE, bias, süre)
- `validation_results.csv`: 3 aday pipeline'ın ve baseline'ın doğrulama MAE, RMSE ve süreleri
- `final_pipeline.json`: nihai pipeline ve baseline parametreleri
- `test_predictions_locked.csv`: test referansları okunmadan kaydedilen baseline ve nihai tahminler (ayrı commit)
- `test_results.csv`: 15 test karesinin gerçek sayıları, tahminleri ve mutlak hataları
- `results/`: ara sonuçlar (veri özellikleri, referans sayılar, parametre taramaları, test özet metrikleri, zamansal değişim, hata analizi)
- `figures/`: hücre görüntüsü içermeyen grafikler (histogramlar, parametre taramaları, deney MAE grafiği, akış diyagramı, zamansal grafikler)

Hücre görüntüsü içeren grafikler `figures_private` klasörüne kaydediliyor. Bu klasör repoya eklenmiyor ve bu grafikler notebook içine gömülmüyor; sadece raporda kullanılıyor.

Not: Süre değerleri bilgisayara göre biraz değişebilir, diğer bütün sonuçlar aynı çıkıyor.
