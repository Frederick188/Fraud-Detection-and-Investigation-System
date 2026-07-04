import requests
import os
from dotenv import load_dotenv

load_dotenv()

IPGEO_API_KEY = os.getenv("IPGEO_API_KEY")
CURRENCY_API_KEY = os.getenv("CURRENCY_API_KEY")

def get_ip_location(ip="8.8.8.8"):
    try:
        url = f"https://api.ipgeolocation.io/ipgeo?apiKey={IPGEO_API_KEY}&ip={ip}"
        r = requests.get(url)
        data = r.json()
        return data.get("country_name", "Unknown"), data.get("city", "Unknown")
    except Exception as e:
        print("Error fetching IP location:", e)
        return "Unknown", "Unknown"


def get_exchange_rate(base="USD", target="INR"):
    try:
        url = f"https://api.currencyfreaks.com/latest?apikey={CURRENCY_API_KEY}&symbols={target}&base={base}"
        r = requests.get(url)
        rate = r.json()["rates"][target]
        return float(rate)
    except Exception as e:
        print("Error fetching exchange rate:", e)
        return 1.0
