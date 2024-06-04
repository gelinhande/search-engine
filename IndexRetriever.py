from pymongo import MongoClient


class IndexRetriever:
    def __init__(self):
        # MongoDB database connection info
        HOST = 'localhost'
        PORT = 27017
        DATABASE_NAME = 'index_database'
        COLLECTION_NAME = 'inverted_index'

        # Create MongoDB client
        self.client = MongoClient(HOST, PORT)

        # Select database and collection
        self.db = self.client[DATABASE_NAME]
        self.collection = self.db[COLLECTION_NAME]

    # Function to retrieve postings for a specific token
    def retrieve_postings_for_token(self, token):
        document = self.collection.find_one({'token': token})
        if document:
            return document['postings']
        else:
            return []

    # Function to retrieve all tokens
    def retrieve_all_tokens(self):
        documents = self.collection.find()
        for document in documents:
            print(f"Token: {document['token']}, Postings: {document['postings']}")


if __name__ == '__main__':
    z = IndexRetriever()
    token_to_search = 'example'  # Assume this is the token you want to search for
    print(z.retrieve_postings_for_token(token_to_search))  # Retrieve postings for a specific token
    # index_retriever.retrieve_all_tokens()  # Retrieve all tokens
