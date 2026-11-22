"""Görüntü işleme adımları ve pipeline'ı çalıştıran fonksiyon.

Bir pipeline bir sözlükle tanımlanıyor:
{
  "yogunluk": {"yontem": "orijinal" | "kontrast_germe" | "hist_esitleme" | "clahe" | "hist_eslestirme", ...},
  "filtre":   {"yontem": "yok" | "gauss" | "median" | "keskinlestirme", "kernel": 5, ...},
  "esikleme": {"yontem": "sabit" | "otsu" | "adaptif", ...},
  "morfoloji": [["acma", 3], ["kapama", 5], ...]   # sırayla uygulanır, sayı = disk yarıçapı
  "delik_doldurma": true/false,
  "min_alan": 150,                                  # pikselden küçük nesneler atılır
  "ayirma": {"yontem": "yok" | "watershed", "min_mesafe": 15}
}
"""
import time

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import exposure
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import disk
from skimage.segmentation import watershed

# ----------------------------------------------------------------------------- yoğunluk dönüşümleri
_HM_REFERANS = {}  # histogram eşleştirme için referans görüntü (sadece geliştirme kümesinden)


def histogram_referansi_ayarla(img):
    _HM_REFERANS["img"] = img


def yogunluk_donusumu(img, ayar):
    y = ayar["yontem"]
    if y == "orijinal":
        return img
    if y == "kontrast_germe":
        alt, ust = np.percentile(img, (ayar.get("alt_yuzdelik", 1), ayar.get("ust_yuzdelik", 99.5)))
        return exposure.rescale_intensity(img, in_range=(alt, ust), out_range=(0, 255)).astype(np.uint8)
    if y == "hist_esitleme":
        return cv2.equalizeHist(img)
    if y == "clahe":
        clahe = cv2.createCLAHE(clipLimit=ayar.get("clip", 2.0), tileGridSize=(ayar.get("kutu", 8),) * 2)
        return clahe.apply(img)
    if y == "hist_eslestirme":
        ref = _HM_REFERANS["img"]
        return np.clip(exposure.match_histograms(img, ref), 0, 255).astype(np.uint8)
    raise ValueError(y)


# ----------------------------------------------------------------------------- filtreler
KESKINLESTIRME_KERNELI = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)


def filtre_uygula(img, ayar):
    f = ayar["yontem"]
    if f == "yok":
        return img
    if f == "gauss":
        k = ayar["kernel"]
        return cv2.GaussianBlur(img, (k, k), 0)
    if f == "median":
        return cv2.medianBlur(img, ayar["kernel"])
    if f == "keskinlestirme":
        # önce hafif yumuşatma (gürültü büyümesin), sonra 3x3 keskinleştirme konvolüsyonu
        on = cv2.GaussianBlur(img, (ayar.get("on_gauss", 5),) * 2, 0)
        return cv2.filter2D(on, -1, KESKINLESTIRME_KERNELI)
    raise ValueError(f)


# ----------------------------------------------------------------------------- eşikleme
def esikle(img, ayar):
    e = ayar["yontem"]
    if e == "sabit":
        return img > ayar["esik"]
    if e == "otsu":
        return img > threshold_otsu(img)
    if e == "adaptif":
        blok, c = ayar["blok"], ayar["C"]
        m = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blok, c)
        m = m > 0
        # karanlık arka planda gürültü çıkmasın diye çok koyu pikseller hücre sayılmıyor
        return m & (img > ayar.get("min_parlaklik", 0))
    raise ValueError(e)


# ----------------------------------------------------------------------------- morfoloji
def morfoloji_uygula(maske, adimlar):
    m = maske.astype(np.uint8)
    for islem, r in adimlar:
        se = disk(r).astype(np.uint8)
        if islem == "erozyon":
            m = cv2.erode(m, se)
        elif islem == "genisletme":
            m = cv2.dilate(m, se)
        elif islem == "acma":
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, se)
        elif islem == "kapama":
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se)
        else:
            raise ValueError(islem)
    return m.astype(bool)


def kucukleri_sil(maske, min_alan):
    lab, n = ndi.label(maske)
    if n == 0:
        return maske
    alanlar = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    tut = np.zeros(n + 1, bool)
    tut[1:] = alanlar >= min_alan
    return tut[lab]


def watershed_ayir(maske, min_mesafe):
    """Distance transform + watershed ile birbirine değen hücreleri ayırır."""
    mesafe = ndi.distance_transform_edt(maske)
    lab_m, _ = ndi.label(maske)
    tepeler = peak_local_max(mesafe, min_distance=min_mesafe, labels=lab_m, exclude_border=False)
    isaret = np.zeros(maske.shape, np.int32)
    isaret[tuple(tepeler.T)] = np.arange(1, len(tepeler) + 1)
    return watershed(-mesafe, isaret, mask=maske)


# ----------------------------------------------------------------------------- pipeline
def pipeline_calistir(img, p, ara_adimlar=False):
    """Görüntüyü pipeline'dan geçirir, etiket görüntüsünü ve hücre sayısını döndürür."""
    ara = {"orijinal": img}
    x = yogunluk_donusumu(img, p["yogunluk"]); ara["yogunluk"] = x
    x = filtre_uygula(x, p["filtre"]); ara["filtre"] = x
    m = esikle(x, p["esikleme"]); ara["esikleme"] = m
    m = morfoloji_uygula(m, p.get("morfoloji", [])); ara["morfoloji"] = m
    if p.get("delik_doldurma", False):
        m = ndi.binary_fill_holes(m)
    m = kucukleri_sil(m, p.get("min_alan", 0)); ara["temiz_maske"] = m
    ayirma = p.get("ayirma", {"yontem": "yok"})
    if ayirma["yontem"] == "watershed":
        etiket = watershed_ayir(m, ayirma["min_mesafe"])
        # watershed sonrası min_alan'dan küçük parçalar hücre sayılmıyor
        alanlar = np.bincount(etiket.ravel())
        kucuk = np.where(alanlar < p.get("min_alan", 0))[0]
        etiket[np.isin(etiket, kucuk[kucuk > 0])] = 0
    else:
        etiket = label(m, connectivity=2)
    sayi = int(len(np.unique(etiket)) - 1)
    ara["etiket"] = etiket
    return (etiket, sayi, ara) if ara_adimlar else (etiket, sayi)


def zamanli_say(img, p):
    t0 = time.perf_counter()
    _, sayi = pipeline_calistir(img, p)
    return sayi, time.perf_counter() - t0
