from Projects.NewspaperArticleSummarizer.summarize import summarize
from fetch_article import fetch_article

article = fetch_article(
    'https://www.theguardian.com/technology/2023/jul/12/claude-2-anthropic-launches-chatbot-rival-chatgpt')

print("Summarizing: " + article['title'])

summary = summarize(article)

print(summary)
