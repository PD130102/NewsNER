import streamlit as st
import pandas as pd
import os
import numpy as np
import sys
from decimal import *
import codecs
import requests

st.title("News-NER")
st.subheader("Named Entity Recognition for news articles")
st.sidebar.title("News-NER")
st.sidebar.write("NewsNER is a Information Extraction tool for news articles.")

tag_set = set()
word_set = set()


def load_data():
    df_hindi = pd.read_csv("./dataset/hi-train.csv")
    return df_hindi


def preprocessed_data():
    df_hindi = pd.read_csv("./dataset/preprocessed_text.csv")
    return df_hindi


def POSTag():
    def read_and_display_text_file(file_path):
        with open(file_path, "r") as file:
            content = file.read()
            st.text_area("POS Tagged Data", content, height=300)
            return content
    file_path = "/Users/pranavdeepak/Pranav Deepak/Ed-Work/University/NewsNER/hmmoutput.txt"
    return read_and_display_text_file(file_path)


def posTagLive(user_input):
    def parse_traindata():
        word_set = set()
        fin = "hmmmodel.txt"
        output_file = "hmmmoutput_live.txt"
        transition_prob = {}
        emission_prob = {}
        tag_list = []
        tag_count = {}
        global tag_set
        try:
            input_file = codecs.open(fin, mode="r", encoding="utf-8")
            lines = input_file.readlines()
            flag = False

            for line in lines:
                line = line.strip("\n")

                if line != "Emission Model":
                    i = line[::-1]

                    key_insert = line[: -i.find(":") - 1]

                    value_insert = line.split(":")[-1]

                    # for transition probabilities #

                    if flag == False:
                        transition_prob[key_insert] = value_insert

                        if (key_insert.split("~tag~")[0] not in tag_list) and (
                            key_insert.split("~tag~")[0] != "start"
                        ):
                            tag_list.append(key_insert.split("~tag~")[0])

                    else:
                        # for emission probabilities #
                        emission_prob[key_insert] = value_insert
                        val = key_insert.split("/")[-1]
                        j = key_insert[::-1]
                        word = key_insert[: -j.find("/") - 1].lower()
                        word_set.add(word)
                        if val in tag_count:
                            tag_count[val] += 1
                        else:
                            tag_count[val] = 1
                        tag_set.add(val)

                else:
                    flag = True
                    continue

            input_file.close()
            return tag_list, transition_prob, emission_prob, tag_count, word_set

        except IOError:
            fo = codecs.open(output_file, mode="w", encoding="utf-8")
            fo.write("File not found: {}".format(fin))
            fo.close()
            sys.exit()

    def viterbi_algorithm(
        sentence, tag_list, transition_prob, emission_prob, tag_count, word_set
    ):
        print(tag_count)

        global tag_set
        # Get words from each sentence #
        sentence = sentence.strip("\n")
        word_list = sentence.split(" ")
        current_prob = {}
        for tag in tag_list:
            # transition probability #
            tp = Decimal(0)
            # Emission probability #
            em = Decimal(0)
            # Storing the probability of every tag to be starting tag #
            if "start~tag~" + tag in transition_prob:
                tp = Decimal(transition_prob["start~tag~" + tag])
            # Check for word in training data. If present, check the probability of the first word to be of given tag#

            if word_list[0].lower() in word_set:
                if (word_list[0].lower() + "/" + tag) in emission_prob:
                    em = Decimal(emission_prob[word_list[0].lower() + "/" + tag])
                    # Storing probability of current combination of tp and em #
                    current_prob[tag] = tp * em

            # Check for word in training data. If absent then probability is just tp#
            else:
                em = Decimal(1) / (tag_count[tag] + len(word_set))
                current_prob[tag] = tp

        if len(word_list) == 1:
            # Return max path if only one word in sentence #
            max_path = max(current_prob, key=current_prob.get)
            return max_path
        else:
            # Tracking from second word to last word #
            for i in range(1, len(word_list)):
                previous_prob = current_prob
                current_prob = {}
                locals()["dict{}".format(i)] = {}
                previous_tag = ""
                for tag in tag_list:
                    if word_list[i].lower() in word_set:
                        if word_list[i].lower() + "/" + tag in emission_prob:
                            em = Decimal(
                                emission_prob[word_list[i].lower() + "/" + tag]
                            )
                            # Find the maximum probability using previous node's(tp*em)[i.e probability of reaching to the previous node] * tp * em (Bigram Model) #

                            max_prob, previous_state = max(
                                (
                                    Decimal(previous_prob[previous_tag])
                                    * Decimal(
                                        transition_prob[previous_tag + "~tag~" + tag]
                                    )
                                    * em,
                                    previous_tag,
                                )
                                for previous_tag in previous_prob
                            )

                            current_prob[tag] = max_prob
                            locals()["dict{}".format(i)][
                                previous_state + "~" + tag
                            ] = max_prob
                            previous_tag = previous_state
                    else:
                        em = Decimal(1) / (tag_count[tag] + len(word_set))

                        max_prob, previous_state = max(
                            (
                                Decimal(previous_prob[previous_tag])
                                * Decimal(transition_prob[previous_tag + "~tag~" + tag])
                                * em,
                                previous_tag,
                            )
                            for previous_tag in previous_prob
                        )

                        current_prob[tag] = max_prob
                        locals()["dict{}".format(i)][
                            previous_state + "~" + tag
                        ] = max_prob
                        previous_tag = previous_state

                # if last word of sentence, then return path dicts of all words #
                if i == len(word_list) - 1:
                    max_path = ""
                    last_tag = max(current_prob, key=current_prob.get)
                    max_path = max_path + last_tag + " " + previous_tag
                    for j in range(len(word_list) - 1, 0, -1):
                        for key in locals()["dict{}".format(j)]:
                            data = key.split("~")
                            if data[-1] == previous_tag:
                                max_path = max_path + " " + data[0]
                                previous_tag = data[0]
                                break
                    result = max_path.split()
                    result.reverse()
                    return " ".join(result)

    tag_list, transition_model, emission_model, tag_count, word_set = parse_traindata()
    fin = "hmminput_live.txt"
    with open(fin, "w") as file:
        file.write(user_input)

    input_file = codecs.open(fin, mode="r", encoding="utf-8")
    fout = codecs.open("hmmoutput_live.txt", mode="w", encoding="utf-8")
    for sentence in input_file.readlines():
        path = viterbi_algorithm(
            sentence, tag_list, transition_model, emission_model, tag_count, word_set
        )
        sentence = sentence.strip("\n")
        word = sentence.split(" ")
        tag = path.split(" ")
        for j in range(0, len(word)):
            if j == len(word) - 1:
                fout.write(word[j] + "/" + tag[j] + "\n")
            else:
                fout.write(word[j] + "/" + tag[j] + " ")

def classify_MNB(input):
    url = "http://127.0.0.1:8000/get_text_classification/MNB"
    payload = {"sentence": input}
    headers = {"Content-Type": "application/json"}
    response = requests.request("POST", url, headers=headers, json=payload)
    return response.text

def classify_LLM(input):
    url = "http://127.0.0.1:8000/get_text_classification/LLM"
    payload = {"sentence": input}
    headers = {"Content-Type": "application/json"}
    response = requests.request("POST", url, headers=headers, json=payload)
    return response.text

def text_summarizer(input):
    url = "http://127.0.0.1:8000/get_text_summarization"
    payload = {"sentence": input}
    headers = {"Content-Type": "application/json"}
    response = requests.request("POST", url, headers=headers, json=payload)
    return response.text

def predict_NER(input):
    url = "http://127.0.0.1:8000/get_ner_tags"
    payload = {"sentence": input}
    headers = {"Content-Type": "application/json"}
    response = requests.request("POST", url, headers=headers, json=payload)
    return response.text


if __name__ == "__main__":
    dataset_name = st.sidebar.selectbox(
        "Select Dataset", ["Select", "NewsNER"]
    )

    if dataset_name == "NewsNER":
        st.sidebar.subheader("Options")
        option = st.sidebar.radio(
            "Choose an Option",
            [
                "Select",
                "View the Dataset",
                "Preprocessed Data",
                "POS Tagged Data",
                "Input POS Tagging",
                "Classficiation",
                "NER",
                "Text Summarization"
            ],
        )

        if option == "View the Dataset":
            df_hindi = load_data()
            st.write(df_hindi)

        if option == "Preprocessed Data":
            df_hindi = preprocessed_data()
            st.write(df_hindi)

        if option == "POS Tagged Data":
            POSTag()

        if option == "Input POS Tagging":
            user_input = st.text_input("Enter the Hindi Sentence")
            posTagLive(user_input)
            with open("hmmoutput_live.txt", "r") as file:
                st.write(file.read())

        if option == "Classficiation":
            st.write("Classficiation")
            classifier_name = st.selectbox(
                "Select Classifier",
                ["Select", "Multinominal Naive Bayes", "LLM"],
            )
            if classifier_name == "Multinominal Naive Bayes":
                st.write("Selected Classifier: ", classifier_name)
                user_input = st.text_input("Enter the Hindi Sentence")
                if st.button("Classify"):
                    st.write("Output")
                    st.write(classify_MNB(user_input))
            elif classifier_name == "LLM":
                st.write("Selected Classifier: ", classifier_name)
                user_input = st.text_input("Enter the Hindi Sentence")
                if st.button("Classify"):
                    st.write("Output")
                    st.write(classify_LLM(user_input))
            else:
                st.write("Select a Classifier")
        if option == "NER":
            st.write("NER")
            user_input = st.text_input("Enter the Hindi Sentence")
            if st.button("Predict"):
                st.write("Output")
                st.write(predict_NER(user_input))
        if option == "Text Summarization":
            st.write("Text Summarization")
            user_input = st.text_input("Enter the Hindi Sentence")
            if st.button("Summarize"):
                st.write("Output")
                st.write(text_summarizer(user_input))
    else:
        st.write("Select a Dataset")

