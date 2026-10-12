import io
import math
import struct
import time
import wave
import requests

def main():
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        samples = []
        for i in range(16000 * 3):
            t = float(i) / 16000
            val = int(32767.0 * 0.3 * math.sin(2.0 * math.pi * 440.0 * t))
            samples.append(struct.pack('<h', val))
        f.writeframes(b''.join(samples))
    wav_data = buf.getvalue()

    print("1. Submitting test_e2e_verification.wav to POST http://127.0.0.1:8000/api/v1/analyze...")
    files = {'file': ('test_e2e_verification.wav', wav_data, 'audio/wav')}
    data = {'num_speakers': 1}
    res = requests.post('http://127.0.0.1:8000/api/v1/analyze', files=files, data=data)
    print("Response status:", res.status_code)
    job_info = res.json()
    print("Response payload:", job_info)
    job_id = job_info['job_id']

    print(f"\n2. Polling GET http://127.0.0.1:8000/api/v1/jobs/{job_id}...")
    for poll_idx in range(40):
        time.sleep(2)
        status_res = requests.get(f'http://127.0.0.1:8000/api/v1/jobs/{job_id}')
        status_json = status_res.json()
        current_st = status_json.get("status")
        created_at = status_json.get("created_at")
        print(f"[{poll_idx+1}] Job status: {current_st} (created_at={created_at})")
        if current_st in ('COMPLETED', 'FAILED'):
            print("\n3. SUCCESS! Final Job Record:")
            print("Job ID:", status_json.get("job_id"))
            print("Status:", status_json.get("status"))
            print("File name:", status_json.get("file_name"))
            print("Created at:", status_json.get("created_at"))
            print("Utterances count:", len(status_json.get("utterances", [])))
            return

    print("Polling timed out!")

if __name__ == "__main__":
    main()
