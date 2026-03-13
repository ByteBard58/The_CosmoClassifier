import requests
from bs4 import BeautifulSoup
import datetime

TARGET_URL = "https://cosmo-classifier.onrender.com/"

def health_checker(target_url=TARGET_URL):
  timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
  try:
    response = requests.get(target_url,timeout=10)
    soup = BeautifulSoup(response.text,"html.parser")
    title = soup.find("title").text if soup.find("title") else "NO TITLE"

    print(f"[{timestamp}] Status:{response.status_code} | Title:'{title}' | Time elapsed:{response.elapsed.total_seconds():.2f}s")
    return {"status":response.status_code, "title":title, "ok":True, "error":None}
  except Exception as e:
    print(f"[{timestamp}] ERROR: {str(e)}")
    return {"status":None, "title": None, "ok":False, "error":str(e)}

if __name__ == "__main__":
  health_checker()
