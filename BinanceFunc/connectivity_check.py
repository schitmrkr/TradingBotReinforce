import requests

base_url = "https://fapi.binance.com"

resp = requests.get(base_url + "/fapi/v3/time", verify=False)

print(resp.json())

