import time
from collections import deque
import requests

#### GESTION DES TRADUCTIONS ####

_traduction_items = None
_traduction_unites = None
_traduction_synergies = None

_items_vers_api = None
_unites_vers_api = None
_synergies_vers_api = None

def _charger_dictionnaires():
    global _traduction_items, _traduction_unites, _traduction_synergies
    global _items_vers_api, _unites_vers_api, _synergies_vers_api

    if _traduction_items is not None:
        return

    url_cdragon = "https://raw.communitydragon.org/latest/cdragon/tft/fr_fr.json"

    try:
        reponse = requests.get(url_cdragon)
        reponse.raise_for_status()
        dico_brut = reponse.json()
    except requests.RequestException as e:
        print(f"Erreur lors du téléchargement de CDragon : {e}")
        _traduction_items = {}
        _traduction_unites = {}
        _traduction_synergies = {}
        _items_vers_api = {}
        _unites_vers_api = {}
        _synergies_vers_api = {}
        return

    _traduction_items = {
        item['apiName']: (item['name'].strip() if item['name'] else item['apiName'])
        for item in dico_brut.get('items', [])
    }

    _traduction_unites = {}
    _traduction_synergies = {}

    set_data = dico_brut.get('sets', {}).get('16', {})

    for champion in set_data.get('champions', []):
        _traduction_unites[champion['apiName']] = champion['name'].strip()

    for trait in set_data.get('traits', []):
        _traduction_synergies[trait['apiName']] = trait['name'].strip()

    _items_vers_api = {v.strip(): k for k, v in _traduction_items.items()}
    _unites_vers_api = {v.strip(): k for k, v in _traduction_unites.items()}
    _synergies_vers_api = {v.strip(): k for k, v in _traduction_synergies.items()}

def traduire_item(api_name):
    _charger_dictionnaires()
    return _traduction_items.get(api_name.strip(), api_name)

def traduire_champion(api_name):
    _charger_dictionnaires()
    return _traduction_unites.get(api_name.strip(), api_name)

def traduire_synergie(api_name):
    _charger_dictionnaires()
    return _traduction_synergies.get(api_name.strip(), api_name)

def api_item(nom):
    _charger_dictionnaires()
    return _items_vers_api.get(nom.strip(), nom)

def api_champion(nom):
    _charger_dictionnaires()
    return _unites_vers_api.get(nom.strip(), nom)

def api_synergie(nom):
    _charger_dictionnaires()
    return _synergies_vers_api.get(nom.strip(), nom)

#### GESTION DE L'API RIOT ####

class RateLimiter:
    def __init__(self, apikey):
        # (max_requests, window_in_seconds)
        self.limits = [
            (20, 1),  # 20 req/1s
            (100, 120)  # 100 req/120s
        ]
        self.requests = deque()
        self.headers = {"X-Riot-Token": apikey}

    def _clean_old_requests(self, now):
        max_window = max(limit[1] for limit in self.limits)
        while self.requests and self.requests[0] <= now - max_window:
            self.requests.popleft()

    def can_send(self) -> bool:
        now = time.time()
        self._clean_old_requests(now)
        for max_reqs, window in self.limits:
            count = sum(1 for ts in self.requests if ts > now - window)
            if count >= max_reqs:
                return False
        return True

    def wait_until_ready(self):
        """Blocks execution until a request slot is available."""
        while True:
            now = time.time()
            self._clean_old_requests(now)

            wait_times = []
            for max_reqs, window in self.limits:
                # Filter requests within this specific window
                current_window_reqs = [ts for ts in self.requests if ts > now - window]

                if len(current_window_reqs) >= max_reqs:
                    # The oldest request in this window must expire before we can send again
                    oldest_in_window = current_window_reqs[0]
                    wait_times.append(oldest_in_window + window - now)

            if not wait_times:
                return  # Ready to send

            # Sleep for the longest required wait time (plus a tiny buffer)
            wait = max(wait_times)
            print(f"waiting {wait} for next api request")
            time.sleep(wait + 0.1)

    def request(self, url):
        self.wait_until_ready()
        self.requests.append(time.time())
        return requests.get(url, headers=self.headers)