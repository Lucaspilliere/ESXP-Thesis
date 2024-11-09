import requests
from bs4 import BeautifulSoup
import pandas as pd
from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

##########################
##       SCRAPPING      ##
##########################

date_list = []
titre_list = []
body_list = []

cac40 = ["cac-40-FR0003500008","FR0003500008/actualites/cac-40","^FCHI"]
nvidia = ["nvidia-US67066G1040","US67066G1040/actualites/nvidia","NVDA"]
tte = ["totalenergies-FR0000120271","FR0000120271/actualites/totalenergies","TTE.PA"]
tencent = ["tencent-holdings-ltd-KYG875721634","KYG875721634/actualites/tencent-holdings-ltd","NNND.BE"]
VIX = "^VIX"

for page in range(1,2):
    url = "https://www.tradingsat.com/"+tte[0]+"/actualites-"+str(page)+".html" 
    headers = {"User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'}
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    for dates in soup.findAll('div', class_='meta-date'):
        title_dates = dates.get_text(strip=True)  # Extract article date
        date_list.append(title_dates)
    
    for link in soup.findAll('a'):
        lien=link.get('href')
        if tte[1] in lien: #TO CHANGE
            new_url = "https://www.tradingsat.com"+lien
            page = requests.get(new_url, headers=headers)
            soup = BeautifulSoup(page.content, 'html.parser')
    
            for title in soup.findAll('h1', class_='h1-article'): # Extract article title
                title_text = title.get_text(strip=True) 
                titre_list.append(title_text)
                
            for body in soup.findAll('div', class_='article-body'): # Extract article body
                body_text = body.get_text(strip=True)
                body_list.append(body_text)
            
# Create the pandas dataframe
df = pd.DataFrame({'Date': date_list,'Titre': titre_list, 'Body': body_list})

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(title):
        score = analyzer.polarity_scores(title)
        return score['compound']
    
# Sentiment score
df['Sentiment Score Body'] = df['Body'].apply(get_sentiment_score)
df['Sentiment Score Title'] = df['Titre'].apply(get_sentiment_score)
df['Sentiment Score'] = df[['Sentiment Score Body','Sentiment Score Title']].mean(axis=1)


# Custom function to convert the Date column
def convert_date(date_str):
    today = datetime.today()
    
    # If the format is "12h45", assume today's date with the given time
    if 'h' in date_str:
        return datetime.strptime(f"{today.strftime('%Y-%m-%d')} {date_str}", '%Y-%m-%d %Hh%M')
    
    # If the date is "Yesterday"
    elif date_str == "Hier":
        return today - timedelta(days=1)
    
    # If the format is "dd/mm/yy", we handle it by recognizing the 2-digit year format
    elif len(date_str.split('/')) == 3:  # This is for 'dd/mm/yy' format
        return pd.to_datetime(date_str, format='%d/%m/%y')  # Use %y for two-digit year
    
    # For "dd/mm" format, assume it's from the current year
    else:
        year = today.year
        return pd.to_datetime(f"{date_str}/{year}", format='%d/%m/%Y')

# Apply the custom function to the Date column
df['Date'] = df['Date'].apply(convert_date)

# Monthly average sentiment score :
df['YearMonth'] = df['Date'].dt.to_period('M')
monthly_avg_df = df.groupby('YearMonth')['Sentiment Score'].mean().reset_index()
monthly_avg_df.columns = ['Month', 'Average Sentiment Score']
monthly_avg_df = monthly_avg_df.sort_values('Month').reset_index(drop=True)
print(monthly_avg_df)


##########################
##         GRAPH        ##
##########################

import matplotlib.pyplot as plt
import seaborn as sns

# Configurate graph style
sns.set(style="whitegrid")

# Histogram sentiment score
plt.figure(figsize=(12, 6))
sns.histplot(df['Sentiment Score'], bins=30, kde=True, color='blue')
plt.title('Sentiment Score Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.axvline(x=0, color='red', linestyle='--', label='Neutre (0)')
plt.legend()
plt.show()

# Dot plot sentiment score
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Sentiment Score'], marker='o', linestyle='-', color='purple')
plt.title('Sentiment Score over time')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.xticks(rotation=45)
plt.axhline(y=0, color='red', linestyle='--', label='Neutre (0)')
plt.legend()
plt.tight_layout()
plt.show()

##########################
##       ANALYSIS       ##
##########################


import statsmodels.api as sm
import yfinance as yf

# New df with date as index
start_date = df['Date'].min().date()
end_date = pd.to_datetime("today").date() 

df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# Return stocks
ticker = tte[2] #TO CHANGE
data = yf.download(ticker, start=start_date, end=end_date)
data['Return'] = data['Adj Close'].pct_change()
returns_df = data[['Return']].dropna()

# Merged df (returns and sentiment score)
merged_df = returns_df.merge(df, left_index=True, right_index=True, how='left')

# Merged df without NaN values
merged_df = merged_df.dropna(subset=['Sentiment Score', 'Return'])

# Regression
X = merged_df[['Sentiment Score']].dropna()
y = merged_df['Return'].dropna()
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())






        