import random
import threading
import time

import requests

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/140.0 Safari/537.36 (academic research crawler; contact via project page)")
COOKIES = {"birthtime": "0", "lastagecheckage": "1-0-1990", "wants_mature_content": "1",
           "Steam_Language": "english"}


class RateLimiter:
    def __init__(self, rps):
        self.interval = 1.0 / rps
        self.lock = threading.Lock()
        self.next_t = time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.next_t:
                time.sleep(self.next_t - now)
            self.next_t = max(now, self.next_t) + self.interval


class SteamSession:
    def __init__(self, rps=2.0, timeout=30):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
        self.s.cookies.update(COOKIES)
        self.rl = RateLimiter(rps)
        self.timeout = timeout

    def get(self, url, params=None, max_tries=8):
        backoff = 5.0
        for attempt in range(max_tries):
            self.rl.wait()
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
            except requests.RequestException:
                time.sleep(backoff + random.random())
                backoff = min(backoff * 2, 300)
                continue
            if r.status_code == 200:
                return r
            if r.status_code in (429, 403, 500, 502, 503, 504):
                time.sleep(backoff + random.random())
                backoff = min(backoff * 2, 300)
                continue
            return r
        return None
