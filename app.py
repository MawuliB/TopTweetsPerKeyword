import streamlit as st
import streamlit.components.v1 as comp
from streamlit_option_menu import option_menu as option
import requests
import pandas as pd

import tweepy

# Import the wordcloud library
import wordcloud

import matplotlib.pyplot as plt

import regex as re
import nltk
import string
from emoji import demojize

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import geocoder

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

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


selected = option(
    menu_title=None,
    options=["Search Trends", "Search Tweets", "Contact"],
    orientation="horizontal",
)

if selected == "Search Trends":

    def get_trends(api, loc):
        # Object that has location's latitude and longitude.
        g = geocoder.osm(loc)

        closest_loc = api.closest_trends(g.lat, g.lng)
        trends = api.get_place_trends(closest_loc[0]["woeid"])
        return trends[0]["trends"][0:20]

    st.title("Get Top Latest Trends per Country")
    st.write("Know what is going!")

    p_qry = ""

    qry = st.text_input("Enter the Country")

    if st.button("Search") or (p_qry != qry):
        p_qry = qry
        loc = qry
        trends = get_trends(api, loc)

        for t in trends:
            st.write(t["name"])


if selected == "Search Tweets":

    selected = option(
        menu_title=None, options=["Tweets", "Statistics"], orientation="horizontal",
    )
    if selected == "Tweets":
        st.title("Get Top Latest Tweets About A Hashtag Or Keyword")
        st.write("Know what is going")

        prev_qry = ""

        query = st.text_input("Enter the Keyword or Hashtag")

        pnum = ""

        num = st.text_input("Enter Number of tweets.")
        st.write("The more tweets the longer it takes. 1000 is recommended")

        if (st.button("Search") or (prev_qry != query)) and (pnum != num):
            prev_qry = query
            pnum = num

            keyword = query
            limit = int(num)
            trends = []

            tweets = tweepy.Cursor(
                api.search_tweets,
                q=keyword,
                count=100,
                tweet_mode="extended",
                lang="en",
            ).items(limit)

            trends.extend(tweets)

            print("Tweet Fetched")

            lists = []

            for t in trends:
                lists.append([t.full_text, f"https://twitter.com/user/status/{t.id}"])

            df = pd.DataFrame(lists)

            columns = ["Tweet_Text", "Link"]
            df.columns = columns

            # df.to_csv("tweets.csv", index=False)

            # df = pd.read_csv("tweets.csv")

            df = df.drop_duplicates(subset="Tweet_Text")
            # st.write(df.describe())

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
                    re.sub(
                        "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", data
                    ).split()
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
                # table = str.maketrans("", "", string.punctuation)
                # data = data.translate(table)

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

            # Load the library with the CountVectorizer method

            # Helper function
            def plot_10_most_common_words(count_data, count_vectorizer):

                words = count_vectorizer.get_feature_names()
                total_counts = np.zeros(len(words))
                for t in count_data:
                    total_counts += t.toarray()[0]

                count_dict = zip(words, total_counts)
                count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
                words = [w[0] for w in count_dict]

                return words

            # Initialise the count vectorizer with the English stop words
            count_vectorizer = CountVectorizer(stop_words="english")

            # Fit and transform the processed titles
            count_data = count_vectorizer.fit_transform(df["processed"])

            # Visualise the 10 most common words
            top_words = plot_10_most_common_words(count_data, count_vectorizer)

            text_l = " ".join(top_words)

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
                .head(20)["Link"]
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
                comp.html(text, height=700)

    if selected == "Statistics":
        st.title("Get Statistics About A Hashtag Or Keyword")
        st.write("Know what is going")

        prev_qry = ""

        query = st.text_input("Enter the Keyword or Hashtag")

        pnum = ""

        num = st.text_input("Enter Number of tweets.")
        st.write("The more tweets the longer it takes. 1000 is recommended")

        if (st.button("Search") or (prev_qry != query)) and (pnum != num):
            prev_qry = query
            pnum = num

            keyword = query
            limit = int(num)
            trends = []

            tweets = tweepy.Cursor(
                api.search_tweets,
                q=keyword,
                count=100,
                tweet_mode="extended",
                lang="en",
            ).items(limit)

            trends.extend(tweets)

            print("Tweet Fetched")

            lists = []

            for t in trends:
                lists.append([t.full_text, f"https://twitter.com/user/status/{t.id}"])

            df = pd.DataFrame(lists)

            columns = ["Tweet_Text", "Link"]
            df.columns = columns

            # df.to_csv("tweets.csv", index=False)

            # df = pd.read_csv("tweets.csv")

            df = df.drop_duplicates(subset="Tweet_Text")
            st.write("Tweet Description After Duplicates Removed")
            st.dataframe(pd.DataFrame(df.astype(str).describe().T))
            st.write("Tweets Dataframe")
            st.dataframe(df)

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
                    re.sub(
                        "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", data
                    ).split()
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
                # table = str.maketrans("", "", string.punctuation)
                # data = data.translate(table)

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

            # Join the different processed titles together.
            long_string = " ".join(df["processed"])

            # Create a WordCloud object
            wordcloud = wordcloud.WordCloud()

            # Generate a word cloud
            wordcloud.generate(long_string)

            # Visualize the word cloud
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(wordcloud)
            plt.axis("off")
            st.write("Most Words")
            st.pyplot(fig)
            # st.image(wordcloud.to_image())

            # Load the library with the CountVectorizer method

            # Helper function
            # def plot_10_most_common_words(count_data, count_vectorizer):

            #     words = count_vectorizer.get_feature_names_out()
            #     total_counts = np.zeros(len(words))
            #     for t in count_data:
            #         total_counts += t.toarray()[0]

            #     count_dict = zip(words, total_counts)
            #     count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
            #     words = [w[0] for w in count_dict]
            #     counts = [w[1] for w in count_dict]
            #     x_pos = np.arange(len(words))

            #     plt.bar(x_pos, counts, align="center")
            #     plt.xticks(x_pos, words, rotation=90)
            #     plt.xlabel("words")
            #     plt.ylabel("counts")
            #     plt.title("10 most common words")

            #     return words

            # # Initialise the count vectorizer with the English stop words
            # count_vectorizer = CountVectorizer(stop_words="english")

            # # Fit and transform the processed titles
            # count_data = count_vectorizer.fit_transform(df["processed"])

            # # Visualise the 10 most common words
            # top_words = plot_10_most_common_words(count_data, count_vectorizer)


if selected == "Contact":
    st.title("Here is My Contact Details")

    st.write("Email: \nmawulibadassou5@gmail.com")
    st.write("Phone: \n+233244065972")
    st.markdown('<p> Github <a href="https://github.com/MawuliB" style="text-decoration: none;"> Go to My Github </a> </p>', unsafe_allow_html=True
    )
    st.markdown('<p> LinkedIn <a href="https://linkedin.com/in/mawuli-badassou-8a3021225/" style="text-decoration: none;"> Go to My linkedIn </a> </p>',unsafe_allow_html=True
    )
    st.markdown('<p> Youtube <a href="https://youtube.com/Mawuli" style="text-decoration: none;"> Go to My Youtube </a> </p>', unsafe_allow_html=True
    )
