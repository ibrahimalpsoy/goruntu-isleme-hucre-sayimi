"""Deney tanımları, değerlendirme ve metrikler."""
import copy
import time

import numpy as np
import pandas as pd

from .islemler import pipeline_calistir


# ----------------------------------------------------------------------------- metrikler
def metrikler(tahmin, gercek, sureler=None):
    tahmin, gercek = np.asarray(tahmin, float), np.asarray(gercek, float)
    hata = tahmin - gercek
    sonuc = {
        "MAE": float(np.abs(hata).mean()),
        "RMSE": float(np.sqrt((hata ** 2).mean())),
        "Bias": float(hata.mean()),
        "Exact_count_acc": float((hata == 0).mean()),
    }
    if sureler is not None:
        sonuc["Ort_sure_s"] = float(np.mean(sureler))
    return sonuc


def say(goruntuler, p):
    """Her görüntü için sayım ve süre. goruntuler: {kare: img}"""
    sayilar, sureler = {}, {}
    for k, img in goruntuler.items():
        t0 = time.perf_counter()
        _, s = pipeline_calistir(img, p)
        sureler[k] = time.perf_counter() - t0
        sayilar[k] = s
    return sayilar, sureler


def degerlendir(goruntuler, referans, p):
    sayilar, sureler = say(goruntuler, p)
    kareler = list(goruntuler)
    m = metrikler([sayilar[k] for k in kareler], [referans[k] for k in kareler],
                  [sureler[k] for k in kareler])
    return m, sayilar, sureler


# ----------------------------------------------------------------------------- tanımlar
def baseline(min_alan):
    """Baseline: gri görüntü, Otsu eşikleme, bağlı bileşen analizi, sabit min alan."""
    return {"yogunluk": {"yontem": "orijinal"}, "filtre": {"yontem": "yok"},
            "esikleme": {"yontem": "otsu"}, "morfoloji": [], "delik_doldurma": False,
            "min_alan": min_alan, "ayirma": {"yontem": "yok"}}


# Kontrollü deneylerde başlangıç noktası (E01). Her deney bundan tek bir şeyi değiştiriyor.
REFERANS = {
    "yogunluk": {"yontem": "kontrast_germe", "alt_yuzdelik": 1, "ust_yuzdelik": 99.5},
    "filtre": {"yontem": "gauss", "kernel": 5},
    "esikleme": {"yontem": "adaptif", "blok": 201, "C": -2, "min_parlaklik": 3},
    "morfoloji": [["acma", 3]],
    "delik_doldurma": True,
    "min_alan": 150,
    "ayirma": {"yontem": "yok"},
}


def _degistir(**kw):
    p = copy.deepcopy(REFERANS)
    p.update(copy.deepcopy(kw))
    return p


def deney_listesi(sabit_esik_E11):
    """(deney_id, grup, degisiklik_aciklamasi, pipeline)"""
    D = [
        ("E01", "referans", "Referans konfigürasyon", REFERANS),
        ("E02", "yoğunluk", "Yoğunluk dönüşümü yok (orijinal)", _degistir(yogunluk={"yontem": "orijinal"})),
        ("E03", "yoğunluk", "Histogram eşitleme", _degistir(yogunluk={"yontem": "hist_esitleme"})),
        ("E04", "yoğunluk", "CLAHE (clip 2.0, 8x8)", _degistir(yogunluk={"yontem": "clahe", "clip": 2.0, "kutu": 8})),
        ("E05", "yoğunluk", "Histogram eşleştirme (referans: geliştirme karesi)", _degistir(yogunluk={"yontem": "hist_eslestirme"})),
        ("E06", "filtre", "Filtre yok", _degistir(filtre={"yontem": "yok"})),
        ("E07", "filtre", "Gaussian 3x3", _degistir(filtre={"yontem": "gauss", "kernel": 3})),
        ("E08", "filtre", "Gaussian 7x7", _degistir(filtre={"yontem": "gauss", "kernel": 7})),
        ("E09", "filtre", "Median 5x5", _degistir(filtre={"yontem": "median", "kernel": 5})),
        ("E10", "filtre", "Keskinleştirme (Gauss 5x5 + 3x3 Laplace kerneli)", _degistir(filtre={"yontem": "keskinlestirme", "on_gauss": 5})),
        ("E11", "segmentasyon", f"Sabit eşik (T={sabit_esik_E11})", _degistir(esikleme={"yontem": "sabit", "esik": sabit_esik_E11})),
        ("E12", "segmentasyon", "Otsu eşikleme", _degistir(esikleme={"yontem": "otsu"})),
        ("E13", "segmentasyon", "Adaptif, blok 101", _degistir(esikleme={"yontem": "adaptif", "blok": 101, "C": -2, "min_parlaklik": 3})),
        ("E14", "morfoloji", "Morfoloji yok", _degistir(morfoloji=[])),
        ("E15", "morfoloji", "Sadece erozyon r=3", _degistir(morfoloji=[["erozyon", 3]])),
        ("E16", "morfoloji", "Sadece genişletme r=3", _degistir(morfoloji=[["genisletme", 3]])),
        ("E17", "morfoloji", "Açma r=5 (büyük yapı elemanı)", _degistir(morfoloji=[["acma", 5]])),
        ("E18", "morfoloji", "Sadece kapama r=3", _degistir(morfoloji=[["kapama", 3]])),
        ("E19", "morfoloji", "Sıra: açma r=3 sonra kapama r=3", _degistir(morfoloji=[["acma", 3], ["kapama", 3]])),
        ("E20", "morfoloji", "Sıra: kapama r=3 sonra açma r=3", _degistir(morfoloji=[["kapama", 3], ["acma", 3]])),
        ("E21", "ayırma", "Distance transform + watershed (min mesafe 25)", _degistir(ayirma={"yontem": "watershed", "min_mesafe": 25})),
        ("E22", "temizlik", "Minimum alan 300", _degistir(min_alan=300)),
    ]
    return D


def ozet_metin(p):
    """Tablo için kısa yazımlar."""
    y = p["yogunluk"]["yontem"]
    y = {"orijinal": "Orijinal", "kontrast_germe": "Kontrast germe", "hist_esitleme": "Hist. eşitleme",
         "clahe": "CLAHE", "hist_eslestirme": "Hist. eşleştirme"}[y]
    f = p["filtre"]
    f = {"yok": "Yok", "gauss": f"Gauss {f.get('kernel')}x{f.get('kernel')}",
         "median": f"Median {f.get('kernel')}x{f.get('kernel')}", "keskinlestirme": "Keskinleştirme"}[f["yontem"]]
    e = p["esikleme"]
    e = {"sabit": f"Sabit T={e.get('esik')}", "otsu": "Otsu",
         "adaptif": f"Adaptif b={e.get('blok')} C={e.get('C')}"}[e["yontem"]]
    ad = {"acma": "açma", "kapama": "kapama", "erozyon": "erozyon", "genisletme": "genişletme"}
    mo = " > ".join(f"{ad[i]} r{r}" for i, r in p["morfoloji"]) or "Yok"
    if p.get("delik_doldurma"):
        mo += " + delik dold."
    if p.get("ayirma", {}).get("yontem") == "watershed":
        mo += f" + watershed ({p['ayirma']['min_mesafe']})"
    par = f"min alan {p['min_alan']}"
    return y, f, e, mo, par
