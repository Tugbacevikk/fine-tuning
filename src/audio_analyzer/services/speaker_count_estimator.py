import logging
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


class SpeakerCountEstimator:
    """
    Kayıt bazlı dinamik konuşmacı sayısı tahmin motoru (Dynamic Speaker Count Estimator).
    
    Sabit kümeleme eşiğine (cluster_threshold) alternatif olarak; ses segmentlerinin
    embedding vektörleri üzerinden k = 1..K_MAX aralığında en optimal küme sayısını belirler.
    
    Temel Prensipler:
    1. Kısa Segment Filtresi: min_segment_sec (varsayılan 1.0s) altındaki kararsız/kısa
       segmentler embedding gürültüsünü önlemek için tahmine dahil edilmez.
    2. k = 2..K_MAX için ortalama bağlantılı Agglomerative Clustering (Cosine Distance) yapılır.
    3. Silhouette Skoru ve Küme Merkezleri (Centroids) arasındaki minimum mesafe hesaplanır.
    4. k=1 ile k>=2 Ayrımı (Silhouette k=1'de tanımsız olduğu için):
       - k=2 küme merkezleri arasındaki kosinüs mesafesi < single_speaker_centroid_dist (varsayılan 0.18) ise:
         İki küme arasındaki benzerlik > %82 demektir. Bu durum iki farklı konuşmacıyı değil,
         tek bir konuşmacının doğal ses/vurgu varyasyonunu gösterir -> k=1 seçilir.
       - max(s(k)) < min_silhouette_threshold (varsayılan 0.22) ise:
         Ayrım zayıftır ve tek bir Gaussian dağılımının yapay bölünmesidir -> k=1 seçilir.
    5. k >= 2 Seçimi:
       - Silhouette skorunu maksimize eden veya Eigengap düşüşünü yakalayan optimal k seçilir.
    """

    def __init__(
        self,
        k_max: int | None = None,
        min_segment_sec: float | None = None,
        single_speaker_centroid_dist: float | None = None,
        min_silhouette_threshold: float | None = None,
        min_distinct_centroid_dist: float | None = None,
        method: str | None = None,
    ):
        from audio_analyzer.config import get_settings
        settings = get_settings()

        self.k_max = k_max if k_max is not None else settings.speaker_est_k_max
        self.min_segment_sec = (
            min_segment_sec if min_segment_sec is not None else settings.speaker_est_min_segment_sec
        )
        self.single_speaker_centroid_dist = (
            single_speaker_centroid_dist
            if single_speaker_centroid_dist is not None
            else settings.speaker_est_centroid_dist_thresh
        )
        self.min_silhouette_threshold = (
            min_silhouette_threshold
            if min_silhouette_threshold is not None
            else settings.speaker_est_min_silhouette
        )
        self.min_distinct_centroid_dist = (
            min_distinct_centroid_dist
            if min_distinct_centroid_dist is not None
            else settings.speaker_est_distinct_centroid_dist
        )
        self.method = (method or settings.speaker_estimation_method).lower()

    def estimate_from_embeddings(
        self,
        embeddings: np.ndarray,
        segment_durations: list[float] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """
        Segment embedding dizisinden (N x D) optimal konuşmacı sayısını tahmin eder.
        
        Args:
            embeddings: N x D float32 normalize embedding matrisi
            segment_durations: Her segmentin saniye cinsinden süresi (opsiyonel)
            
        Returns:
            (estimated_k, details_dict)
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score

        N = len(embeddings)
        details: dict[str, Any] = {
            "num_segments": N,
            "k_scores": {},
            "method": self.method,
            "decision_rule": "default",
        }

        # Yetersiz segment durumu
        if N <= 1:
            details["decision_rule"] = "insufficient_segments"
            return 1, details

        # Kosinüs mesafe matrisi: D = clip(1 - dot(X, X.T), 0, 2)
        S = np.dot(embeddings, embeddings.T)
        D = np.clip(1.0 - S, 0.0, 2.0)

        max_k_possible = min(self.k_max, N - 1)
        if max_k_possible < 2:
            details["decision_rule"] = "single_pair"
            # 2 segment var; aralarındaki mesafe eşiğin altındaysa 1, üstündeyse 2
            pair_dist = float(D[0, 1])
            details["pair_distance"] = pair_dist
            if pair_dist < self.single_speaker_centroid_dist:
                return 1, details
            return 2, details

        # Eigengap analizi (Laplacian)
        A = np.clip(S, 0.0, 1.0)
        deg = np.sum(A, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-8)))
        L_sym = np.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt
        evals = np.sort(np.linalg.eigvalsh(L_sym))
        gaps = np.diff(evals[: max_k_possible + 1])
        details["eigenvalues"] = [round(float(e), 4) for e in evals[: max_k_possible + 1]]
        details["eigengaps"] = [round(float(g), 4) for g in gaps]

        # k = 2..max_k_possible kümeleme denemesi
        k_results = {}
        for k in range(2, max_k_possible + 1):
            clu = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
            labels = clu.fit_predict(D)

            if len(set(labels)) < 2:
                continue

            sil = float(silhouette_score(D, labels, metric="precomputed"))

            # Küme merkezleri ve merkezler arası minimum kosinüs mesafesi
            centroids = []
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    c_vec = np.mean(embeddings[mask], axis=0)
                    norm = np.linalg.norm(c_vec)
                    centroids.append(c_vec / norm if norm > 1e-12 else c_vec)

            c_dists = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    c_dists.append(float(1.0 - np.dot(centroids[i], centroids[j])))

            min_cd = min(c_dists) if c_dists else 0.0
            counts = [int(cnt) for cnt in np.bincount(labels)]

            k_results[k] = {
                "silhouette": round(sil, 4),
                "min_centroid_dist": round(min_cd, 4),
                "cluster_counts": counts,
            }

        details["k_scores"] = k_results

        if not k_results:
            return 1, details

        # -------------------------------------------------------------
        # ADIM 4: k=1 İLE k>=2 ARASINDAKİ AYRIM KURALLARI
        # -------------------------------------------------------------
        k2_info = k_results.get(2)
        if k2_info is not None:
            k2_min_cd = k2_info["min_centroid_dist"]

            # Kural 4a: İki küme merkezi arasındaki kosinüs mesafesi çok düşükse (< 0.18),
            # iki küme aslında aynı kişinin sesidir (benzerlik > %82).
            if k2_min_cd < self.single_speaker_centroid_dist:
                logger.info(
                    "SpeakerCountEstimator: k=2 merkez mesafesi (%.3f < %.3f) tek konuşmacıya işaret ediyor -> k=1",
                    k2_min_cd,
                    self.single_speaker_centroid_dist,
                )
                details["decision_rule"] = "k1_centroid_distance_below_threshold"
                return 1, details

            # Kural 4b: Maksimum silhouette skoru çok düşükse (< 0.22)
            max_sil = max(v["silhouette"] for v in k_results.values())
            if max_sil < self.min_silhouette_threshold:
                logger.info(
                    "SpeakerCountEstimator: Maksimum silhouette skoru (%.3f < %.3f) çok zayıf -> k=1",
                    max_sil,
                    self.min_silhouette_threshold,
                )
                details["decision_rule"] = "k1_silhouette_floor"
                return 1, details

        # -------------------------------------------------------------
        # ADIM 5: k >= 2 SEÇİMİ
        # -------------------------------------------------------------
        # Sadece merkezler arası mesafesi geçerli olan adayları dikkate al
        valid_candidates = {
            k: v for k, v in k_results.items() if v["min_centroid_dist"] >= self.min_distinct_centroid_dist
        }
        if not valid_candidates:
            valid_candidates = k_results

        if self.method == "eigengap" and len(gaps) >= 2:
            # Eigengap yöntemi: ilk 2..K aralığında en büyük fark
            best_k = int(np.argmax(gaps[1:]) + 2)
            details["decision_rule"] = "eigengap_max"
            return best_k, details
        else:
            # Silhouette yöntemi: valid adaylar içinde en yüksek silhouette
            best_k = max(valid_candidates.keys(), key=lambda k: valid_candidates[k]["silhouette"])
            details["decision_rule"] = "silhouette_max"
            return best_k, details
