import streamlit as st
import streamlit.components.v1 as comp
import requests
import pandas as pd

import tweepy
import pandas as pd

st.title("Get Top Latest Tweets About A Hashtag Or Hashtag")
st.write("Know what is going")

prev_qry = ""

query = st.text_input("Enter the Keyword or Hashtag")

if st.button("Search") or (prev_qry != query):
    prev_qry = query
    # Display search results for user_query
    api_key = "yZ6E6wCJh5BpYD6hLSy07aesi"
    api_key_secret = "s98RUSMsrwWNdzkhxMp2a1Jhawy2CYxLgHrFAEl8Itc3JbaGbN"

    access_token = "1082021464986054657-BxSzIQDe1YPef2WPIsU5bijWbs5NBW"
    access_token_secret = "KfvBonYBQOArHZJkQ0vmcfmDIGne1QJf3LuRLucBY9uVe"

    # authentication
    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth, wait_on_rate_limit=True)

    try:
        api.verify_credentials()
        print("Authentication OK")
    except Exception:
        print("Error during authentication")

    print("Successfully connected to the Twitter API.")

    keyword = query
    limit = 10000
    trends = []

    tweets = tweepy.Cursor(
        api.search_tweets, q=keyword, count=100, tweet_mode="extended", lang="en"
    ).items(limit)

    trends.extend(tweets)

    print("Tweet Fetched")

    lists = []

    for t in trends:
        lists.append([t.full_text, f"https://twitter.com/user/status/{t.id}"])

    tweet_df = pd.DataFrame(lists)

    columns = ["Tweet_Text", "Link"]
    tweet_df.columns = columns

    columns = ["Tweet_Text", "Link"]
    tweet_df.columns = columns
    tweet_df.head()

    tweet_df.to_csv("tweets.csv", index=False)

    df = pd.read_csv("tweets.csv")

    df = df.drop_duplicates(subset="Tweet_Text")
    # st.write(df.describe())

    import regex as re
    import nltk
    import string
    from emoji import demojize

    # Some basic helper functions to clean text by removing urls, html tags, punctuations and Stop Words.

    def helper(data):

        # lower_text
        # print(data)
        data = str(data).lower()

        def tr(n):
            if n[0:2] == "rt":
                n = n[3:]
            return n

        data = tr(data)
        data = " ".join(
            re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", data).split()
        )

        # remove_URL
        url = re.compile(r"https?://\S+|www\.\S+")
        data = url.sub(r"", data)

        # remove_html
        html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
        data = re.sub(html, "", data)

        # transform_short_negation_form
        data = data.replace(r"(can't|cannot)", "can not")
        data = data.replace(r"n't", " not")

        # remove_punct
        table = str.maketrans("", "", string.punctuation)
        data = data.translate(table)

        # remove_special
        data = demojize(data)
        data = data.replace(r"::", ": :")
        data = data.replace(r"’", "'")
        data = data.replace(r"[^a-z\':_]", " ")

        # remove_repetitions
        pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
        return data.replace(str(pattern), r"\1")

    df["processed"] = df["Tweet_Text"].apply(lambda x: helper(x))

    df.head()

    # Import the wordcloud library
    import wordcloud

    # Join the different processed titles together.
    long_string = " ".join(df["processed"])

    # Create a WordCloud object
    wordcloud = wordcloud.WordCloud()

    # Generate a word cloud
    wordcloud.generate(long_string)

    # Visualize the word cloud
    st.write(wordcloud.to_image())

    # Load the library with the CountVectorizer method
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    # Helper function
    def plot_10_most_common_words(count_data, count_vectorizer):
        import matplotlib.pyplot as plt

        words = count_vectorizer.get_feature_names()
        total_counts = np.zeros(len(words))
        for t in count_data:
            total_counts += t.toarray()[0]

        count_dict = zip(words, total_counts)
        count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
        words = [w[0] for w in count_dict]
        counts = [w[1] for w in count_dict]
        x_pos = np.arange(len(words))

        plt.bar(x_pos, counts, align="center")
        plt.xticks(x_pos, words, rotation=90)
        plt.xlabel("words")
        plt.ylabel("counts")
        plt.title("10 most common words")
        st.write(plt.show())
        return words

    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer(stop_words="english")

    # Fit and transform the processed titles
    count_data = count_vectorizer.fit_transform(df["processed"])

    # Visualise the 10 most common words
    top_words = plot_10_most_common_words(count_data, count_vectorizer)

    text_l = " ".join(top_words)

    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk
    from nltk import word_tokenize
    from nltk.stem import WordNetLemmatizer

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    stopwords = nltk.corpus.stopwords.words("english")
    extra = [
        "!",
        "(",
        ")",
        "-",
        "[",
        "]",
        "{",
        "}",
        ";",
        ":",
        '"',
        ",",
        "<",
        ">",
        "/",
        "?",
        "@",
        "#",
        "$",
        "%",
        "^",
        "&",
        "*",
        "_",
        "~",
        "'",
    ]

    lemmatizer = WordNetLemmatizer()

    def stopword_tokenize(sentence):
        # tokenize
        word_list = word_tokenize(sentence)
        # remove stop words
        stop_list = [
            word
            for word in word_list
            if word not in stopwords and word not in extra and word.isalpha()
        ]
        # lemmatize result
        final_list = [lemmatizer.lemmatize(word) for word in stop_list]
        return final_list

    vectorizer = TfidfVectorizer(tokenizer=stopword_tokenize)

    def cosine_sim(text1, query):
        tfidf = vectorizer.fit_transform([text1, query])
        return round(((tfidf * tfidf.T).A)[0, 1], 2)

    df["Cosine_sim"] = df["processed"].apply(lambda x: cosine_sim(x, text_l))
    result = (
        df.sort_values(by="Cosine_sim", ascending=False)
        .drop_duplicates()
        .head(10)["Link"]
        .values.tolist()
    )

    def getJson(url):
        link = "https://publish.twitter.com/oembed?url={}".format(url)
        response = requests.get(link)
        json = response.json()["html"]
        return json

    print("Printing Tweets")

    for li in result:
        text = getJson(li)
        comp.html(text, height=500)
