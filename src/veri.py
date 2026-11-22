"""Veri setini okuma ve referans hücre sayıları."""
import zipfile

import cv2
import numpy as np

from . import ayarlar as A


def veriyi_hazirla():
    """Veri klasörü yoksa ve zip varsa zip'i açar."""
    if not A.VERI_KLASORU.exists() and A.ZIP_YOLU.exists():
        with zipfile.ZipFile(A.ZIP_YOLU) as z:
            z.extractall(A.ZIP_YOLU.parent)
    if not A.VERI_KLASORU.exists():
        raise FileNotFoundError(
            "Veri seti bulunamadı. Fluo-N2DH-GOWT1.zip dosyasını data klasörüne koyun (data/README.md).")


def kare_sayisi(sekans):
    return len(sorted((A.VERI_KLASORU / sekans).glob("t*.tif")))


def goruntu_yolu(sekans, kare):
    return A.VERI_KLASORU / sekans / f"t{kare:03d}.tif"


def goruntu_oku(sekans, kare):
    """Gri seviyeli görüntüyü olduğu gibi okur (bu veri setinde uint8)."""
    img = cv2.imread(str(goruntu_yolu(sekans, kare)), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(goruntu_yolu(sekans, kare))
    return img


def tra_oku(sekans, kare):
    return cv2.imread(str(A.VERI_KLASORU / f"{sekans}_GT" / "TRA" / f"man_track{kare:03d}.tif"),
                      cv2.IMREAD_UNCHANGED)


def referans_sayi(sekans, kare):
    """TRA etiket görüntüsünde 0 (arka plan) hariç farklı hücre kimliği sayısı."""
    etiket = tra_oku(sekans, kare)
    kimlikler = np.unique(etiket)
    return int((kimlikler > 0).sum())
