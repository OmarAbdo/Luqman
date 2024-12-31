# Project Development Summary: Sentiment Analysis, Macro-Economic, Fundamental Analysis, and Technical Sentiment Indicators Features

## Sentiment Analysis Feature

- Initially, we considered **VADER** for sentiment analysis due to its efficiency for short texts like tweets. VADER is good at handling slang and emoticons and is lightweight to implement.
- However, for more nuanced sentiment analysis, especially for long-form content like news articles, we have decided to use **GPT-4o Mini**. Despite a higher computational cost compared to VADER, GPT-4o Mini provides richer context understanding and consistent sentiment analysis for both short and long texts.
- Pricing for **GPT-4o Mini** is very cost-effective for MVP implementation, with estimated monthly costs being around **$3.38 for 3,000 articles per month**.

## Macro-Economic Analysis Feature

- We will include macro-economic indicators such as **interest rates, inflation rates, GDP growth, consumer confidence index**, and more. These metrics will be sourced from reliable data providers like **FRED** or **World Bank APIs**.
- Additionally, **sector-specific information** from **Yahoo Finance** will be included as part of the macro-economic analysis, as sector performance significantly impacts a company's stock price.

## Fundamental Analysis Feature

- We have decided on a comprehensive list of fundamental metrics, including **valuation ratios (P/E, P/B), profitability ratios (ROE, ROA), leverage ratios (Debt-to-Equity), liquidity ratios (Current Ratio), growth metrics**, and **cash flow metrics**.
- **Yahoo Finance** will be used as the primary source for these metrics. Any gaps in the data will be filled using **custom calculations**, and as a last resort, we may pull data from **Alpha Vantage**.

## Technical Sentiment Indicators Feature

- We will use **Yahoo Finance** to gather **technical sentiment indicators** like the **VIX (Volatility Index)** and the **Put-Call Ratio**. Yahoo Finance provides **15-minute delayed data**, which is preferred over end-of-day data from services like **Polygon.io** for our short-term trading analysis.
- In the future, as the tool becomes more sophisticated, we will consider using premium data sources and tools for all features, including technical sentiment indicators, to achieve better accuracy and reliability.

## Overall Strategy

- For MVP implementation, we'll focus on getting the **Sentiment Analysis Feature** up and running using **GPT-4o Mini** for both short and long texts.
- We will add **macro-economic indicators** and **fundamental analysis metrics** using a combination of **Yahoo Finance**, **FRED**, and custom calculations to enrich our dataset for the LSTM model.
- **Technical Sentiment Indicators** will also be integrated for a better understanding of broader market sentiment.

We plan to progressively integrate these features while ensuring that each of them is modular, scalable, and aligns with our **short-term trading goals**. Future enhancements include incorporating **manual feature scaling**, **attention mechanisms**, and potentially upgrading sentiment analysis models to more sophisticated LLMs as the project scales.