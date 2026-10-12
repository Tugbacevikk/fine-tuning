"""
Hata Sınıflandırması ve İstisna Hiyerarşisi.
İşlerin geçici (transient - yeniden denenebilir) veya kalıcı (permanent - DLQ/FAILED)
olarak sınıflandırılmasını sağlar.
"""


class JobExecutionError(Exception):
    """Tüm analiz görevi hatalarının temel sınıfı."""



class PermanentJobError(JobExecutionError):
    """
    Kalıcı İş Hatası (Bozuk/desteklenmeyen ses, geçersiz format vb.).
    Yeniden denenmez, doğrudan FAILED durumuna geçer ve DLQ'ya atılır.
    """



class TransientJobError(JobExecutionError):
    """
    Geçici İş Hatası (OOM, Ağ/I/O kesintisi, Zaman aşımı, Storage hatası vb.).
    Maksimum deneme sayısına (MAX_JOB_ATTEMPTS) kadar üstel backoff ile yeniden denenir.
    """



def is_transient_error(ex: Exception) -> bool:
    """
    Verilen istisnanın geçici mi yoksa kalıcı mı olduğunu belirler.
    True -> Geçici (Retry)
    False -> Kalıcı (Direct FAILED / DLQ)
    """
    if isinstance(ex, PermanentJobError):
        return False
    if isinstance(ex, TransientJobError):
        return True

    # Bilinen kalıcı Python istisnaları (bozuk girdi, tip hatası, bulunamayan/silinmiş dosya)
    if isinstance(ex, (ValueError, TypeError, FileNotFoundError)):
        return False

    err_str = str(ex).lower()
    permanent_keywords = [
        "corrupt",
        "unsupported format",
        "invalid audio",
        "cannot decode",
        "header corrupt",
        "not a valid audio",
        "bozuk",
        "desteklenmeyen",
        "file not found",
        "no such file",
        "bulunamadı",
    ]
    if any(kw in err_str for kw in permanent_keywords):
        return False

    # Varsayılan olarak beklenmeyen sistem, ağ, bellek, I/O hataları geçici kabul edilir
    return True
