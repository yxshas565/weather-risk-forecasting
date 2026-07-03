"""
Normalizes non-English country name transliterations found in the raw
dataset (e.g. "Saudi Arabien" -> "Saudi Arabia") so spatial analysis
output is clean. Re-saves cleaned_weather.csv with the fix applied.
"""
import pandas as pd

df = pd.read_csv('data/cleaned_weather.csv')

print(f"Unique countries before fix: {df['country'].nunique()}")

# Mapping built from inspecting the actual non-English entries found in
# script 08's output (German/Portuguese/French/Turkish/Russian spellings)
country_fixes = {
    "Saudi Arabien": "Saudi Arabia",
    "Marrocos": "Morocco",
    "Turkménistan": "Turkmenistan",
    "Турция": "Turkey",
    "Deutschland": "Germany",
    "Österreich": "Austria",
    "Schweiz": "Switzerland",
    "España": "Spain",
    "Italia": "Italy",
    "Brasil": "Brazil",
    "Российская Федерация": "Russia",
    "中国": "China",
    "日本": "Japan",
    "Việt Nam": "Vietnam",
    "المملكة العربية السعودية": "Saudi Arabia",
    "Polska": "Poland",
    "Portugal": "Portugal",
    "México": "Mexico",
    "Nederland": "Netherlands",
    "Norge": "Norway",
    "Sverige": "Sweden",
    "Suomi": "Finland",
    "Danmark": "Denmark",
    "Ελλάδα": "Greece",
    "Türkiye": "Turkey",
    "Éire": "Ireland",
    "Magyarország": "Hungary",
    "România": "Romania",
    "Україна": "Ukraine",
    "Malásia": "Malaysia",
    "كولومبيا": "Colombia",
    "Гватемала": "Guatemala",
    "Польша": "Poland",
    "Polônia": "Poland",
    "Südkorea": "South Korea",
    "Bélgica": "Belgium",
    "火鸡": "Turkey", 
    "Inde": "India",
    "Estonie": "Estonia",
    "Jemen": "Yemen",
    "Komoren": "Comoros",
    "Kyrghyzstan": "Kyrgyzstan",
    "Letonia": "Latvia",
    "Mexique": "Mexico",
    "Saint-Vincent-et-les-Grenadines": "Saint Vincent and the Grenadines",
    "USA United States of America": "United States of America",
    "United Statesof America": "United States of America",
}

df['country'] = df['country'].replace(country_fixes)

print(f"Unique countries after fix: {df['country'].nunique()}")
print(f"Rows changed: {df['country'].isin(country_fixes.values()).sum()}")

df.to_csv('data/cleaned_weather.csv', index=False)
print("Saved data/cleaned_weather.csv with normalized country names.")

# Re-check for any remaining non-ASCII country names we might have missed
remaining_non_ascii = df[df['country'].apply(lambda x: not x.isascii())]['country'].unique()
if len(remaining_non_ascii) > 0:
    print(f"\nWARNING — still non-ASCII country names not covered by the mapping:")
    for c in remaining_non_ascii:
        print(f"  '{c}'")
else:
    print("\nNo remaining non-ASCII country names.")