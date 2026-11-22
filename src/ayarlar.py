"""Projede kullandığım sabitler ve dosya yolları."""
from pathlib import Path

OGRENCI_NO_SON5 = 29040
SEED = OGRENCI_NO_SON5 + 2026  # 31066

KOK = Path(__file__).resolve().parents[1]
VERI_KLASORU = KOK / "data" / "Fluo-N2DH-GOWT1"
ZIP_YOLU = KOK / "data" / "Fluo-N2DH-GOWT1.zip"
SONUC_KLASORU = KOK / "results"
GRAFIK_KLASORU = KOK / "figures"                # veri görüntüsü içermeyen grafikler (repoda)
OZEL_GRAFIK_KLASORU = KOK / "figures_private"   # hücre görüntüsü içeren grafikler (sadece rapor)

# veri seçimi
DEV_SAYISI = 10
VAL_SAYISI = 5
TEST_BLOK_SAYISI = 3
TEST_BLOK_UZUNLUGU = 5
MIN_BOSLUK = 5  # gruplar ve bloklar arasında en az 5 kare boşluk (|fark| >= 6)


def klasorleri_hazirla():
    for k in (SONUC_KLASORU, GRAFIK_KLASORU, OZEL_GRAFIK_KLASORU):
        k.mkdir(parents=True, exist_ok=True)
