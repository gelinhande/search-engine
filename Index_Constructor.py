import json
import os
import pathlib
import re
from collections import defaultdict, Counter
from typing import Generator, Tuple
from nltk.tokenize import RegexpTokenizer
import nltk
import math
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from bs4 import BeautifulSoup
from pymongo import MongoClient

class Index_Construction:
    def __init__(self):
        self.total_file_size = 0
        self.total_number_of_docs = 0
        self.inverted_index = defaultdict(dict)
        self.inverted_info = defaultdict(list)
        self.root_directory = '/Users/lucaszhuang1210gmail.com/Documents/UCI/CS121/Project3_Search_Engine/data/WEBPAGES_RAW'
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client['index_database']
        self.collection = self.db['inverted_index']


        self.STOP_WORDS = {'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are',
                           "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both',
                           'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does',
                           "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further',
                           'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's",
                           'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i',
                           "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its',
                           'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of',
                           'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out',
                           'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't",
                           'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them',
                           'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're",
                           "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was',
                           "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's",
                           'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why',
                           "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've",
                           'your', 'yours', 'yourself', 'yourselves'}

    def constructing_inverted_indexing(self):
        bookkeeping_file_path = os.path.join(self.root_directory, 'bookkeeping.json')
        for id, url in self.generate_bookkeeping_pairs(bookkeeping_file_path):
            file_path = os.path.join(self.root_directory, id.replace('/', os.sep)) # Linux/MacOS/Windows path adjustment
            # print(f"Processing ID: {id} URL: {url}")
            # if not os.path.exists(file_path):
            #     continue  # Skip if file doesn't exist
            # file_path = 'WEBPAGES_RAW/0/157'
            with open(file_path, "r", encoding='utf-8') as page:
                print(file_path)
                self.total_file_size += os.path.getsize(file_path) / 1024

                soup = BeautifulSoup(page.read(), 'html.parser')
                text = soup.get_text().lower()

                if not soup.find() or text.startswith(''):
                    continue  # Skip if no HTML tags found
                lines_split = text.split("\n")  # 将文本按换行符分割成行
                # print(lines_split)
                self.total_number_of_docs += 1
                # one_word_tokens = self.tokenize_lemmatizer(text, 1)
                word_tokens = self.tokenize_lemmatizer(lines_split, 1)
                word_tokens = word_tokens |self.tokenize_lemmatizer(lines_split, 2)
                # print(word_tokens)
                word_tokens_frequency =self.computeWordFrequencies(word_tokens)
                word_tag_dict = self.get_tag_dict(soup)
                # print(word_tokens)
                for token in word_tokens:
                    # Update to include URL in posting
                    if id in self.inverted_index[token]:
                        continue
                    token_importance = self.tag_data_score(token,word_tag_dict)
                    tf_score = 1 + math.log10(word_tokens_frequency[token])

                    self.inverted_index[token] = self.inverted_index[token] | {id: [tf_score, token_importance,
                                                                                    word_tokens[token]]}
                    # print(self.inverted_index)
                    # print(True
                    #       )
                # print('token',token)
                    # print('self.inverted_index[token]',self.inverted_index[token])
                    # self.inverted_info[token].append(url)
                    # print("Token: ", token)
        print("Phase1: Done")

        # 计算tf-idf 并储存
        for token, ids in self.inverted_index.items():
            for id, data in ids.items():
                idf_score = math.log10(self.total_number_of_docs / len(ids))
                self.inverted_index[token][id][0] = data[0] * idf_score





    # Returns a Generator of Tuples
    def generate_bookkeeping_pairs(self, filepath: str) -> Generator[Tuple[str, str], None, None]:
        # A generator function to yield ID-URL pairs from a JSON file.
        with open(filepath, 'r') as file:
            data = json.load(file)
            for key, value in data.items():
                yield key, value


    def computeWordFrequencies(self, token_list):
        word_frequencies = defaultdict(int)
        for token in token_list:
            # Increment the count for each token
            # if token in word_frequencies:
            #     word_frequencies[token] += 1
            # else:
            #     word_frequencies[token] = 1
            word_frequencies[token]+=len(token_list[token])
        return word_frequencies

    def tokenize_lemmatizer(self, lines, n=2,):
        # print(lines)
        n_grams_with_lines = defaultdict(list)

        # tokenizer = RegexpTokenizer('[a-zA-Z0-9]+').tokenize(text)
        tokenizer = RegexpTokenizer('[a-zA-Z0-9]+')
        # print(tokens)
        # tokens = [token for token in tokens if token not in string.punctuation and not re.match(r'^(/|[a-zA-Z]:\\)', token)]
        # 词形还原
        lemmatizer = WordNetLemmatizer()

        for line_number, line in enumerate(lines, start=1):
            tokens = tokenizer.tokenize(line)



            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in self.STOP_WORDS]

        # 生成n-grams
            n_grams = zip(*[lemmatized_tokens[i:] for i in range(n)])
            n_grams = [' '.join(n_gram) for n_gram in n_grams]
            # print(n_grams)
            # print(n_grams)
            for n_gram in n_grams:
                n_grams_with_lines[n_gram].append(line_number)

        return n_grams_with_lines

    def get_tag_dict(self, soup):

        tag_dict = defaultdict(set)
        for title in soup.find_all('title'):
            # print("elf.tokenizelemmatizer(title.get_text())",self.tokenizelemmatizer(title.get_text()))
            tag_dict['title'].update( set(self.tokenizelemmatizer(title.get_text())))
            tag_dict['title'].update( set(self.tokenizelemmatizer(title.get_text(),n=1)))
        # print("tag_dict['title']",tag_dict['title'])

        for bold in soup.find_all('b'):
            tag_dict['bold'].update( set(self.tokenizelemmatizer(bold.get_text())))
            tag_dict['bold'].update( set(self.tokenizelemmatizer(bold.get_text(),n=1)))
        #
        for h1 in soup.find_all('h1'):
            tag_dict['h1'].update( set(self.tokenizelemmatizer(h1.get_text())))
            tag_dict['h1'].update( set(self.tokenizelemmatizer(h1.get_text(),n=1)))
        #
        for h2 in soup.find_all('h2'):
            tag_dict['h2'].update( set(self.tokenizelemmatizer(h2.get_text())))
            tag_dict['h2'].update( set(self.tokenizelemmatizer(h2.get_text(),n=1)))
        #
        for h3 in soup.find_all('h3'):
            tag_dict['h3'].update( set(self.tokenizelemmatizer(h3.get_text())))
            tag_dict['h3'].update( set(self.tokenizelemmatizer(h3.get_text(),n=1)))
        #
        for anchor in soup.find_all('a'):
            tag_dict['anchor'].update( set(self.tokenizelemmatizer(anchor.get_text(),n=1)))
            tag_dict['anchor'].update( set(self.tokenizelemmatizer(anchor.get_text())))

        return tag_dict

    def tag_data_score(self, token, tag_dict):
        tag_score_sum = 0
        if token in tag_dict['title']:

            tag_score_sum += 1
            # print(token)

        if token in tag_dict['h1']:
            tag_score_sum += 0.8

        if token in tag_dict['h2']:
            tag_score_sum += 0.6

        if token in tag_dict['h3']:
            tag_score_sum += 0.4

        if token in tag_dict['anchor']:
            tag_score_sum += 0.3

        if token in tag_dict['bold']:
            tag_score_sum += 0.2

        return tag_score_sum
    # @staticmethod
    # def get_token_positions(lines, token):
    #     s = set(enumerate(lines))
    #     line_numbers = []
    #     for i, line in s:
    #         if token in line:
    #             line_numbers.append(i + 1)  # 记录 token 出现的行号（从1开始）
    #     return line_numbers
    def store_inverted_index_to_mongodb(self):
        documents = []
        for token, postings in self.inverted_index.items():
            # 将每个词条及其对应的posting list转换为MongoDB文档
            document = {
                'token': token,
                'postings': []
            }
            for doc_id, details in postings.items():
                posting = {
                    'doc_id': doc_id,
                    'details': details
                }
                document['postings'].append(posting)
            documents.append(document)

        if documents:
            # 插入文档到MongoDB
            self.collection.insert_many(documents)
            print(f"{len(documents)} documents have been inserted into MongoDB.")
        else:
            print("No documents to insert.")

    def tokenizelemmatizer(self, text, n=2):
        # tokens = word_tokenize(text.lower())
        tokens = RegexpTokenizer('[a-zA-Z0-9]+').tokenize(text.lower())
        # print(tokens)
        # tokens = [token for token in tokens if token not in string.punctuation and not re.match(r'^(/|[a-zA-Z]:\\)', token)]
        # 词形还原
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in self.STOP_WORDS]

        # 生成n-grams
        n_grams = zip(*[lemmatized_tokens[i:] for i in range(n)])
        n_grams = [' '.join(n_gram) for n_gram in n_grams]
        # print(n_grams)

        return n_grams


if __name__ == '__main__':
    # Specify the directory you want to start from
    instance = Index_Construction()
    instance.constructing_inverted_indexing()
    # Number of unique words
    print(len(instance.inverted_index))
    print(len(instance.inverted_info))
    instance.store_inverted_index_to_mongodb()
    # Number of doc