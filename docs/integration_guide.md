# 🏢 Kurumsal Sistem Entegrasyonu & Webhook Kullanım Kılavuzu

Bu doküman, Ses Analizi Platformu'nun üçüncü parti sistemlere (Şehir Hastaneleri HBYS, Çağrı Merkezleri, CRM, ERP) REST API ve **Asenkron Webhook (Geri Bildirim)** ile nasıl entegre edileceğini açıklar.

---

## 1. 🌐 Çevrimdışı (Air-Gapped / On-Premises) Kurulum

Sistem internete hiç bağlanmadan tam çevrimdışı çalışabilir. Modeller sunucuya bir kez indirildikten sonra `HF_HUB_OFFLINE=1` modunda devreye alınır:

```bash
# 1. Modelleri tek seferlik yerel klasöre paketleyin
python scripts/download_offline_models.py

# 2. Çevrimdışı modda sunucuyu başlatın
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WHISPER_MODEL_SIZE=small

uvicorn audio_analyzer.api.main:app --app-dir src --host 0.0.0.0 --port 8000
```

---

## 2. 🔑 API Kimlik Doğrulama & Güvenlik

Sistem `X-API-Key` başlığı üzerinden kimlik doğrulama sağlar. Üretim ortamında (`APP_ENV != development`) `API_KEY` zorunludur.

```http
X-API-Key: your-secret-api-key
```

---

## 3. 📤 Analiz İsteği Gönderme (API Endpoint)

Dış sistemler ses analizi başlatmak için `POST /api/v1/analyze` servisini çağırır.

### İstek Parametreleri (Multipart Form Data):
* `file`: Ses Dosyası (MP3, WAV, FLAC, M4A, OGG)
* `callback_url` *(Opsiyonel)*: Analiz bittiğinde sonucun gönderileceği Webhook adresi (Sadece HTTPS, private IP'ler engellenir).
* `num_speakers` *(Opsiyonel)*: Beklenen konuşmacı sayısı (`1` ile `10` arası tamsayı).
  * `num_speakers=1` gönderildiğinde nöral konuşmacı ayırma (Diarization) adımı **baypas edilir** ve CPU analiz süresi belirgin şekilde hızlanır (Diarization adımı tamamen atlanır).
  * `num_speakers=2` veya üzeri verildiğinde konuşmacı ayırma motoruna hedef konuşmacı kümeleme sayısı enjekte edilir.
* `external_id` *(Opsiyonel)*: Dış sistemdeki benzersiz çağrı ID'si (Idempotency için aynı external_id ile yapılan tekrarlı istekler var olan `job_id`yi döner).

### Örnek cURL İsteği:
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "X-API-Key: your-secret-api-key" \
  -F "file=@/path/to/hastane_cagri_kaydi.wav" \
  -F "num_speakers=1" \
  -F "callback_url=https://hbys.hastane.gov.tr/api/audio-callback"
```

### Dönüş Yanıtı (`HTTP 202 Accepted`):
```json
{
  "job_id": "7d972041-1744-4374-81eb-103a6fa303de",
  "file_name": "hastane_cagri_kaydi.wav",
  "status": "PENDING",
  "message": "Ses dosyası kabul edildi, analiz arka planda başlatıldı."
}
```

---

## 4. 🔔 Webhook (Callback) Geri Bildirim Yapısı

Analiz **COMPLETED** veya **FAILED** durumuna ulaştığında, sistem belirttiğiniz `callback_url` adresine otomatik bir `HTTP POST` bildirimi gönderir.

### Webhook Başlıkları (Headers):
```http
Content-Type: application/json; charset=utf-8
User-Agent: AudioAnalyzer-Webhook/1.0
X-Timestamp: 1775130000
X-Signature: sha256=5a8d7e9f2b1a3c4d... (HMAC-SHA256 İmza Başlığı)
```

---

## 5. 🛡️ HMAC-SHA256 İmza Doğrulama (Güvenlik Kılavuzu)

Webhook çağrılarında imza üretimi şu formülle yapılır:
$$\text{Signature} = \text{"sha256="} + \text{HMAC-SHA256}(\text{WEBHOOK\_SECRET}, \text{timestamp} + \text{"."} + \text{raw\_json\_body})$$

Yeniden oynatma (replay attack) saldırılarını engellemek için alıcı sistemlerin timestamp farkını (maksimum 300 saniye / 5 dakika) kontrol etmesi tavsiye edilir.

### 🐍 Python ile Alıcı Doğrulama Örneği (FastAPI / Flask):

```python
import time
import hmac
import hashlib

def verify_webhook_signature(
    raw_body: bytes,
    received_signature: str,
    timestamp_header: str,
    webhook_secret: str,
    tolerance_seconds: int = 300
) -> bool:
    # 1. Replay attack kontrolü (5 dakika zaman toleransı)
    try:
        req_timestamp = int(timestamp_header)
        if abs(time.time() - req_timestamp) > tolerance_seconds:
            return False  # Zaman aşımı
    except (ValueError, TypeError):
        return False

    # 2. HMAC-SHA256 hesaplama: sha256=HMAC(secret, f"{timestamp}.{body}")
    body_str = raw_body.decode('utf-8')
    expected_data = f"{timestamp_header}.{body_str}".encode('utf-8')
    expected_hash = hmac.new(
        webhook_secret.encode('utf-8'),
        expected_data,
        hashlib.sha256
    ).hexdigest()
    expected_signature = f"sha256={expected_hash}"

    # 3. Sabit zamanlı güvenli karşılaştırma (Timing Attack Koruması)
    return hmac.compare_digest(expected_signature, received_signature)
```

### 🟢 Node.js / Express ile Alıcı Doğrulama Örneği:

```javascript
const crypto = require('crypto');

function verifyWebhookSignature(req, webhookSecret, toleranceSeconds = 300) {
    const signatureHeader = req.headers['x-signature'];
    const timestampHeader = req.headers['x-timestamp'];

    if (!signatureHeader || !timestampHeader) return false;

    // 1. Replay attack kontrolü
    const now = Math.floor(Date.now() / 1000);
    const reqTimestamp = parseInt(timestampHeader, 10);
    if (isNaN(reqTimestamp) || Math.abs(now - reqTimestamp) > toleranceSeconds) {
        return false;
    }

    // 2. HMAC-SHA256 hesaplama
    const rawBody = req.body; // Raw string / Buffer body
    const dataToSign = `${timestampHeader}.${rawBody}`;
    const expectedHash = crypto
        .createHmac('sha256', webhookSecret)
        .update(dataToSign, 'utf8')
        .digest('hex');
    const expectedSignature = `sha256=${expectedHash}`;

    // 3. Güvenli karşılaştırma
    return crypto.timingSafeEqual(
        Buffer.from(expectedSignature),
        Buffer.from(signatureHeader)
    );
}
```
