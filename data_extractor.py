import json
import datetime
from tft_utils import RateLimiter

# Configuration
output_file = "data/dump.json"
API_KEY = "riot api key" # 
REGION = "euw1"
CONTINENT = "europe"

# Patch 16.2 : 8 Janvier 2026 au 20 janvier 2026, on met le 21 pour inclure toute la journée du 20
target_patch = "16.1"
date_debut = datetime.datetime(2026, 1, 8)
date_fin = datetime.datetime(2026, 1, 21)

matches_sauvegardes = [] # Liste qui va contenir nos JSON finaux

print("Initialisation du rate limiter")
limiter = RateLimiter(API_KEY)

print("Récupération des joueurs Challenger")
url_chall = f"https://{REGION}.api.riotgames.com/tft/league/v1/challenger"
data_chall = limiter.request(url_chall).json()

print("Extraction des PUUIDs de chaque joueur")
puuids = [joueur['puuid'] for joueur in data_chall.get('entries', [])]

print("Récupération des matchs")
match_ids_uniques = set()  # Le 'set' empêche d'avoir des doublons si 2 joueurs étaient dans la même game

# Conversion en Timestamps (le format en secondes que Riot utilise)
start_ts = int(date_debut.timestamp())
end_ts = int(date_fin.timestamp())

for puuid in puuids:
    # L'api ne nous permet que de récupérer 200 parties à la fois, raison pour laquelle nous devons faire une boucle pour faire plus de requêtes si nécessaire
    start = 0
    while True:
        url_matches = f"https://{CONTINENT}.api.riotgames.com/tft/match/v1/matches/by-puuid/{puuid}/ids?startTime={start_ts}&endTime={end_ts}&start={start}&count=200"
        response = limiter.request(url_matches)

        if response.status_code != 200:
            print("ERREUR LORS DE LA RECUPERATION DES MATCHS")
            print(response.status_code, response.text)
            break

        match_ids = response.json()
        if not match_ids:
            break
        match_ids_uniques.update(match_ids)
        if len(match_ids) < 200:
            break
        start += 200

# On filtre pour être sûr de n'avoir pris que les parties du patch que l'on veut
match_ids_liste = list(match_ids_uniques)
for match_id in match_ids_liste:
    url_detail = f"https://{CONTINENT}.api.riotgames.com/tft/match/v1/matches/{match_id}"
    response = limiter.request(url_detail)
    if response.status_code != 200:
        print("ERREUR LORS DE LA RECUPERATION DES DONNEES DU MATCH")
        print(response.status_code, response.text)
        continue
    match_data = response.json()
    game_version = match_data['info']['game_version']
    if f"<Releases/{target_patch}>" in game_version:
        matches_sauvegardes.append(match_data)
    else:
        print(f"X Match {match_id} ignoré (Mauvaise version: {game_version})")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(matches_sauvegardes, f, ensure_ascii=False, indent=4)

print(f"Données sauvegardées dans {output_file}.")
