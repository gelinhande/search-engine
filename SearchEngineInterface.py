import json
import tkinter as tk
from tkinter import ttk, scrolledtext
import IndexRetriever
import Index_Constructor



class SearchEngineInterface:
    def __init__(self):
        self.display_range = 15
        self.load_bookkeeping()
        self.create_search_engine_window()
        self.prioritized_search_results = []
        self.index_constructor = Index_Constructor.Index_Construction()
        self.retriever = IndexRetriever.IndexRetriever()


    def load_bookkeeping(self):
        with open('/Users/lucaszhuang1210gmail.com/Documents/UCI/CS121/Project3_Search_Engine/data/WEBPAGES_RAW/bookkeeping.json', 'r') as file:
            self.bookkeeping = json.load(file)

    def search(self, query):
        # Get grams from query
        two_gram_words = self.index_constructor.tokenizelemmatizer(query, n=2)
        one_gram_words = self.index_constructor.tokenizelemmatizer(query, n=1)

        # Find postings that match n_gram words
        two_gram_results = []
        one_gram_results = []
        for bi_word in two_gram_words:
            postings = self.retriever.retrieve_postings_for_token(bi_word)
            two_gram_results.extend(postings)
        for single_word in one_gram_words:
            postings = self.retriever.retrieve_postings_for_token(single_word)
            one_gram_results.extend(postings)

        # Compute Priority, 2 gram match are more likely to be relevant
        prioritized_postings = []
        prioritized_postings.extend(self.calculate_priority(two_gram_results, 2))
        prioritized_postings.extend(self.calculate_priority(one_gram_results, 1))

        # Sort the postings by priority in descending order (highest priority first)
        prioritized_postings.sort(key=lambda x: x['priority'], reverse=True)

        # Extract and return a list of doc_ids based on the sorted priorities
        prioritized_doc_ids = [posting['doc_id'] for posting in prioritized_postings]

        return prioritized_doc_ids


    def calculate_priority(self, postings, amplifier):
        # priority = (tf-idf + html tag score) * amplifier
        prioritized_postings = [{
            'priority': (posting['details'][0] + posting['details'][1]) * amplifier,
            'doc_id': posting['doc_id']
        } for posting in postings]

        return prioritized_postings

    def handle_search(self):
        # Reset display range
        self.display_range = 15
        query = self.search_box.get()
        self.prioritized_search_results = self.search(query)
        self.display_result()


    def display_result(self):
        # Make the results_display writable before inserting text
        self.results_display.config(state=tk.NORMAL)
        self.results_display.delete('1.0', tk.END)  # Clear previous results
        # Display only the first 15 results
        # displayed_results = self.prioritized_search_results[self.display_range-15 : self.display_range]
        for doc_id in self.prioritized_search_results:
            self.results_display.insert(tk.END, self.bookkeeping.get(doc_id) + '\n')

        # Set the results_display back to disabled (read-only) state
        self.results_display.config(state=tk.DISABLED)


    def create_search_engine_window(self):
        self.window = tk.Tk()
        self.window.title("ICS Search Engine")

        search_logo = ttk.Label(self.window, text="ReallySmartSearch")
        search_label = ttk.Label(self.window, text="Enter your search query below:")
        search_logo.pack(pady=(10, 0))
        search_label.pack(pady=(10, 0))

        self.search_box = ttk.Entry(self.window, width=60)
        self.search_box.pack(pady=5)

        search_button = ttk.Button(self.window, text="Search", command=self.handle_search)
        search_button.pack(pady=10)

        self.results_display = scrolledtext.ScrolledText(self.window, width=70, height=15, wrap=tk.WORD)
        self.results_display.pack(pady=10)
        self.results_display.config(state=tk.DISABLED)  # Start as read-only

    def run(self):
        self.window.mainloop()

if __name__ == '__main__':
    search_interface = SearchEngineInterface()
    search_interface.run()
