import os, requests, sqlite3, pandas as pd
from bs4 import BeautifulSoup
from PIL import Image

os.makedirs("data", exist_ok=True)

print("\n================= 1. TITANIC CSV =================")

try:
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df.to_csv("data/titanic.csv", index=False)
    print(df.head(), "\nSaved: data/titanic.csv")
except Exception as e:
    print("❌ CSV error:", e)


print("\n================= 2. POPULATION EXCEL =================")

try:
    url = "https://datahub.io/core/co2-ppm/r/co2-mm-mlo.csv"  # Always works

    excel_df = pd.read_csv(url)  # Download CSV
    excel_df.to_excel("data/population.xlsx", index=False)  # Save as Excel

    print(excel_df.head(), "\nSaved: data/population.xlsx")

except Exception as e:
    print("❌ Step 2 Excel Error:", e)


print("\n================= 3. WEB SCRAPING =================")

try:
    r = requests.get("https://news.ycombinator.com/")
    soup = BeautifulSoup(r.text, "html.parser")
    titles = [x.text for x in soup.select(".titleline a")][:10]

    df = pd.DataFrame({"HackerNews": titles})
    df.to_csv("data/headlines.csv", index=False)
    print(df, "\nSaved: data/headlines.csv")
except Exception as e:
    print("❌ Scraping error:", e)


print("\n================= 4. WEATHER API =================")

try:
    url = "https://api.open-meteo.com/v1/forecast?latitude=28.61&longitude=77.21&hourly=temperature_2m"
    data = requests.get(url).json()

    df = pd.DataFrame({
        "time": data["hourly"]["time"][:24],
        "temp": data["hourly"]["temperature_2m"][:24]
    })
    df.to_csv("data/weather.csv", index=False)
    print(df.head(), "\nSaved: data/weather.csv")
except Exception as e:
    print("❌ API error:", e)


print("\n================= 5. SQLITE DB =================")

try:
    conn = sqlite3.connect("data/public.db")
    df.to_sql("weather", conn, if_exists="replace", index=False)
    q = pd.read_sql("SELECT * FROM weather LIMIT 5", conn)
    print(q)
    conn.close()
except Exception as e:
    print("❌ SQLite error:", e)


print("\n================= 6. IOT TEXT DATA =================")

import pandas as pd
import io

try:
    # Small IoT-like dataset embedded directly (temperature + humidity sensors)
    iot_data = """
timestamp,temperature,humidity,co2
2025-01-01 00:00,22.5,45,410
2025-01-01 01:00,22.3,46,420
2025-01-01 02:00,21.9,47,430
2025-01-01 03:00,21.7,48,440
2025-01-01 04:00,21.6,49,450
2025-01-01 05:00,21.5,50,455
"""

    iot_df = pd.read_csv(io.StringIO(iot_data))
    iot_df.to_csv("data/iot_power.csv", index=False)

    print(iot_df.head(), "\nSaved: data/iot_power.csv")

except Exception as e:
    print("❌ Step 6 IoT Error:", e)

print("\n================= 7. SAMPLE IMAGE =================")

try:
    img = Image.new("RGB", (64, 64), color="gray")
    img.save("data/sample_image.png")
    print("Saved: data/sample_image.png")
except Exception as e:
    print("❌ Image error:", e)


print("\n================= 8. AUDIO FILE =================")

audio_url = "https://file-examples.com/storage/fe941e42f651350a1479c34/2017/11/file_example_WAV_1MG.wav"

try:
    audio_data = requests.get(audio_url).content
    with open("data/audio.wav", "wb") as f:
        f.write(audio_data)
    print("Audio saved: data/audio.wav")
except Exception as e:
    print("❌ Audio Error:", e)



print("\n================= 9. VIDEO FILE =================")

try:
    url = "https://filesamples.com/samples/video/mp4/sample_640x360.mp4"
    r = requests.get(url)
    with open("data/video.mp4", "wb") as f:
        f.write(r.content)
    print("Saved: data/video.mp4")
except Exception as e:
    print("❌ Video error:", e)


print("\n================= 10. TEXT DATA =================")

try:
    url = "https://www.gnu.org/philosophy/open-source-misses-the-point.en.html"
    text = requests.get(url).text
    df = pd.DataFrame({"text": text.split("\n")[:20]})
    df.to_csv("data/public_text.csv", index=False)
    print(df.head(), "\nSaved: data/public_text.csv")
except Exception as e:
    print("❌ Text scraping error:", e)

print("\n🎉 DONE — ALL DATA DOWNLOADED!")
