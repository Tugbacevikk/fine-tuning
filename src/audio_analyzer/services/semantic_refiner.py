import json
import logging
import os
import re
import urllib.request
import uuid

from audio_analyzer.config import get_settings
from audio_analyzer.domain.models import TranscriptUtterance

logger = logging.getLogger(__name__)


class SemanticRefiner:
    """
    Anlamsal Konuşmacı Hizalama ve Cümle Düzeltme Servisi.
    Genel amaçlı ses analizlerinde noktalama ve süre hizalamalarını düzeltir.
    Opsiyonel olarak 'call_center' gibi alan odaklı (domain-specific) kurallarla
    veya yerel Ollama LLM servisiyle gelişmiş konuşmacı ayrıştırmasını destekler.
    """

    def __init__(
        self,
        domain_mode: str | None = None,
        use_llm: bool | None = None,
        ollama_url: str | None = None,
        model_name: str | None = None,
        custom_triggers: dict[str, list[str]] | None = None,
        split_on_questions: bool | None = None,
    ):
        settings = get_settings()
        self.use_llm = use_llm if use_llm is not None else settings.use_llm
        self.ollama_url = ollama_url or settings.ollama_url
        self.model_name = model_name or settings.ollama_model

        if domain_mode is not None:
            logger.warning("domain_mode parametresi ('%s') kullanımdan kaldırılmıştır ve yok sayılmaktadır.", domain_mode)
        if custom_triggers:
            logger.warning("custom_triggers parametresi kullanımdan kaldırılmıştır ve yok sayılmaktadır.")
        if split_on_questions is not None:
            logger.warning("split_on_questions parametresi kullanımdan kaldırılmıştır ve yok sayılmaktadır.")

    def _query_ollama_llm(self, prompt: str) -> str | None:
        """
        Yerel Ollama LLM servisine (Llama-3.2 / Qwen-2.5) istek atarak anlamsal analiz yaptırır.
        Ollama erişilebilir değilse None döner.
        """
        try:
            timeout_sec = float(os.getenv("OLLAMA_TIMEOUT_SEC", "45.0"))
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 256, "temperature": 0.1},
            }
            req = urllib.request.Request(
                self.ollama_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response")
        except Exception:
            return None

    def refine(self, utterances: list[TranscriptUtterance]) -> list[TranscriptUtterance]:
        """
        Utterance listesini anlamsal cümle, hizalama ve tekrar temizliği kontrolüne sokar.
        """
        if not utterances:
            return []

        # 0. Halüsinatif kelime/sözcük grubu tekrarlarını ve imla hatalarını temizle
        cleaned_utterances: list[TranscriptUtterance] = []
        for u in utterances:
            cleaned_txt = self._clean_repetitive_text(u.text)
            cleaned_txt = self._normalize_turkish_text(cleaned_txt)
            cleaned_utterances.append(
                TranscriptUtterance(
                    id=u.id,
                    speaker_id=u.speaker_id,
                    start_time=u.start_time,
                    end_time=u.end_time,
                    text=cleaned_txt,
                )
            )
        utterances = cleaned_utterances

        # 1. LLM Modu aktifse ve PIPELINE_PROFILE != 'feedback' ise LLM adımını çağır
        pipeline_profile = os.getenv("PIPELINE_PROFILE", "full").lower()
        if self.use_llm and pipeline_profile != "feedback":
            utterances = self._refine_with_llm(utterances)

        # 2. Sandviç ve tekil mikro parazit konuşmacıları düzelt
        smoothed = self._smooth_orphan_and_sandwich_speakers(utterances)

        # 3. Aynı konuşmacının kısa aralıklı parçalarını birleştir
        return self._normalize_short_gaps(smoothed)

    def _clean_repetitive_text(self, text: str) -> str:
        """
        Ardışık 3 veya daha fazla olan halüsinatif kelime ve kelime öbeği tekrarlarını temizler.
        "yavaş yavaş", "evet evet", "güzel güzel" gibi 2'li ikilemelere dokunmaz.
        """
        if not text:
            return text

        words = text.strip().split()
        if not words:
            return text

        # 1. Ardışık 3+ kelime öbeği tekrarlarını temizle (örn: "bu sefer bu sefer bu sefer bu sefer" -> "bu sefer")
        if len(words) >= 6:
            new_words = []
            i = 0
            while i < len(words):
                if i + 5 < len(words):
                    p1 = [w.lower() for w in words[i:i+2]]
                    p2 = [w.lower() for w in words[i+2:i+4]]
                    p3 = [w.lower() for w in words[i+4:i+6]]
                    if p1 == p2 == p3:
                        new_words.extend(words[i:i+2])
                        i += 6
                        while i + 1 < len(words) and [w.lower() for w in words[i:i+2]] == p1:
                            i += 2
                        continue
                new_words.append(words[i])
                i += 1
            words = new_words

        # 2. Ardışık 3+ tekil kelimeleri temizle (2'li ikilemelere dokunma)
        if len(words) >= 3:
            new_words = []
            i = 0
            while i < len(words):
                j = i
                while j < len(words) and words[j].lower() == words[i].lower():
                    j += 1
                repeat_count = j - i
                if repeat_count >= 3:
                    new_words.append(words[i])
                else:
                    new_words.extend(words[i:j])
                i = j
            words = new_words

        text = " ".join(words)

        parts = [p.strip() for p in text.split(",") if p.strip()]
        if parts:
            cleaned_parts = []
            i = 0
            while i < len(parts):
                j = i
                while j < len(parts) and parts[j].lower() == parts[i].lower():
                    j += 1
                repeat_count = j - i
                if repeat_count >= 3:
                    cleaned_parts.append(parts[i])
                else:
                    cleaned_parts.extend(parts[i:j])
                i = j
            text = ", ".join(cleaned_parts)

        return text

    def _normalize_turkish_text(self, text: str) -> str:
        """
        Noktalama öncesi boşlukları temizler (örn: "mi ?" -> "mi?").
        Kelimeye veya güne bağlı elle yazılmış kurallar içermez.
        """
        if not text:
            return text

        # Soru işaretleri ve noktalamalardan önceki boşlukları temizle ("mi ?" -> "mi?")
        return re.sub(r'\s+([\?\!\,\.\:\;])', r'\1', text)

    def _refine_with_llm(self, utterances: list[TranscriptUtterance]) -> list[TranscriptUtterance]:
        """
        Yerel Ollama LLM (Llama-3.2 / Qwen-2.5) kullanarak konuşmacı etiketlerini anlamsal bütüne göre düzeltir.
        Ollama yanıt vermezse veya JSON ayrıştırma hatası olursa orijinal konuşmacı listesini güvenle döndürür (fail-open).
        """
        if not utterances:
            return utterances

        logger = logging.getLogger(__name__)

        payload = [
            {"idx": idx, "text": u.text}
            for idx, u in enumerate(utterances)
        ]

        payload_json = json.dumps(payload, ensure_ascii=False)

        prompt = (
            "Sen uzman bir telefon konuşması diyalog ayrıştırma asistanısın.\n"
            "Aşağıdaki diyalog dökümünde yalnızca 2 taraf konuşmaktadır: Temsilci/Görevli (SPEAKER_00) ve Müşteri/Hasta (SPEAKER_01).\n"
            "Her 'idx' cümlesinin Temsilci mi (SPEAKER_00) yoksa Müşteri mi (SPEAKER_01) olduğunu anlamsal içeriğe göre tespit et.\n"
            "ÇIKTI FORMATI: SADECE aşağıdaki JSON formatında geçerli bir JSON listesi döndür. Başka hiçbir açıklama yazma!\n"
            'Örnek: [{"idx": 0, "speaker": "SPEAKER_00"}, {"idx": 1, "speaker": "SPEAKER_00"}, {"idx": 2, "speaker": "SPEAKER_00"}, {"idx": 3, "speaker": "SPEAKER_01"}]\n\n'
            f"GİRDİ CÜMLELERİ:\n{payload_json}"
        )

        raw_response = self._query_ollama_llm(prompt)
        if not raw_response:
            logger.info("Ollama LLM servisine ulaşılamadı veya zaman aşımına uğradı. Orijinal döküm korunuyor.")
            return utterances

        try:
            clean_resp = raw_response.strip()
            if "```json" in clean_resp:
                clean_resp = clean_resp.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_resp:
                clean_resp = clean_resp.split("```")[1].split("```")[0].strip()

            match = re.search(r'\[.*\]', clean_resp, re.DOTALL)
            if match:
                clean_resp = match.group(0)

            parsed_list = json.loads(clean_resp)
            if not isinstance(parsed_list, list):
                return utterances

            spk_map = {}
            for item in parsed_list:
                if isinstance(item, dict) and "idx" in item and "speaker" in item:
                    spk_map[int(item["idx"])] = str(item["speaker"]).strip()

            if not spk_map:
                return utterances

            updated_utterances: list[TranscriptUtterance] = []
            for idx, u in enumerate(utterances):
                new_spk = spk_map.get(idx, u.speaker_id)
                if not new_spk.startswith("SPEAKER_"):
                    new_spk = f"SPEAKER_{new_spk}"
                updated_utterances.append(
                    TranscriptUtterance(
                        id=u.id,
                        speaker_id=new_spk,
                        start_time=u.start_time,
                        end_time=u.end_time,
                        text=u.text,
                    )
                )

            logger.info("Ollama LLM anlamsal konuşmacı düzeltmesi başarıyla uygulandı (%d segment).", len(spk_map))
            return updated_utterances

        except Exception as err:
            logger.warning("Ollama LLM yanıtı ayrıştırılamadı (%s). Orijinal döküm korunuyor.", err)
            return utterances



    def _normalize_short_gaps(
        self, utterances: list[TranscriptUtterance]
    ) -> list[TranscriptUtterance]:
        """
        Aynı konuşmacının 0.6 saniyeden kısa aralıklı parçalanmış cümlelerini ve
        noktalama ile bitmemiş yarım cümleleri (mid-sentence split) tek bir konuşmacı kartında birleştirir.
        """
        if len(utterances) <= 1:
            return utterances

        max_silence_threshold = float(os.getenv("MAX_SILENCE_THRESHOLD", "1.5"))
        max_clause_gap = float(os.getenv("MAX_CLAUSE_GAP", "1.5"))

        merged: list[TranscriptUtterance] = []
        i = 0
        while i < len(utterances):
            curr = utterances[i]
            while i + 1 < len(utterances):
                nxt = utterances[i + 1]
                gap = nxt.start_time - curr.end_time
                curr_text = curr.text.strip()
                nxt_text = nxt.text.strip()

                if not curr_text or not nxt_text:
                    break

                # Birleştirme yalnızca aynı konuşmacı (nxt.speaker_id == curr.speaker_id) ise yapılır
                is_same_speaker = (gap <= max_silence_threshold and nxt.speaker_id == curr.speaker_id)

                curr_is_unpunctuated = not curr_text.endswith((".", "?", "!", ":", ";", "…"))

                is_clause_split = (
                    gap <= max_clause_gap
                    and curr_is_unpunctuated
                    and nxt.speaker_id == curr.speaker_id
                )

                if is_same_speaker or is_clause_split:
                    combined_text = (curr_text + " " + nxt_text).strip()
                    curr = TranscriptUtterance(
                        id=curr.id,
                        speaker_id=curr.speaker_id,
                        start_time=curr.start_time,
                        end_time=nxt.end_time,
                        text=combined_text,
                    )
                    i += 1
                else:
                    break
            merged.append(curr)
            i += 1

        return merged

    def _smooth_orphan_and_sandwich_speakers(
        self, utterances: list[TranscriptUtterance]
    ) -> list[TranscriptUtterance]:
        """
        1. Sandviç Konuşmacı Düzeltmesi (Sandwich Pattern):
           A -> B -> A sıralamasında B'nin süresi kısa (< 0.8s) veya metni az ise
           B konuşmacısı parazit geçiş sayılarak A konuşmacısına geri bağlanır.
        2. Mikro Gürültü / Punctuation-only Segmentler:
           0.35s'den kısa veya yalnızca noktalama (?, ., !) içeren mikro parçaların
           konuşmacı kimliği komşu konuşmacıya devredilir.
        """
        if len(utterances) <= 1:
            return utterances

        from audio_analyzer.config import get_settings
        settings = get_settings()

        result: list[TranscriptUtterance] = [
            TranscriptUtterance(
                id=u.id,
                speaker_id=u.speaker_id,
                start_time=u.start_time,
                end_time=u.end_time,
                text=u.text,
            )
            for u in utterances
        ]

        # 1. Sandviç Geçiş Düzeltmesi (A -> B -> A)
        if settings.enable_sandwich_rule:
            max_dur = settings.sandwich_max_duration_sec
            for i in range(1, len(utterances) - 1):
                # Kararları zincirleme etki yapmaması için ORİJİNAL utterances listesine göre ver:
                orig_prev = utterances[i - 1]
                orig_curr = utterances[i]
                orig_next = utterances[i + 1]

                if orig_prev.speaker_id == orig_next.speaker_id and orig_curr.speaker_id != orig_prev.speaker_id:
                    duration = orig_curr.end_time - orig_curr.start_time
                    # Kelime sayısı koşulu (len(txt_words) <= 3) kaldırıldı; sadece süreye bakılıyor:
                    if duration < max_dur:
                        result[i] = TranscriptUtterance(
                            id=orig_curr.id,
                            speaker_id=orig_prev.speaker_id,
                            start_time=orig_curr.start_time,
                            end_time=orig_curr.end_time,
                            text=orig_curr.text,
                        )

        # 2. Mikro Parazit / Punctuation-only Segment Yumuşatma
        if settings.enable_micro_utterance_rule:
            for i in range(len(utterances)):
                orig_curr = utterances[i]
                duration = orig_curr.end_time - orig_curr.start_time
                clean_txt = orig_curr.text.strip()

                is_pure_punct = clean_txt in ("?", ".", "!", "...", "") or len(clean_txt) <= 2
                is_ultra_short = duration < 0.35

                if is_pure_punct or is_ultra_short:
                    target_spk = None
                    if i > 0:
                        target_spk = utterances[i - 1].speaker_id
                    elif i + 1 < len(utterances):
                        target_spk = utterances[i + 1].speaker_id

                    if target_spk and target_spk != orig_curr.speaker_id:
                        result[i] = TranscriptUtterance(
                            id=orig_curr.id,
                            speaker_id=target_spk,
                            start_time=orig_curr.start_time,
                            end_time=orig_curr.end_time,
                            text=orig_curr.text,
                        )

        return result

