import math

class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self,index,termWeighting):
        self.index = index
        self.termWeighting = termWeighting

    # Method to apply query to index
    def forQuery(self,query):
        if self.termWeighting == 'binary':
            return binary.get_docs(self, query)
        elif self.termWeighting == 'tf':
            return tf.get_tfdocs(self, query)
        elif self.termWeighting == 'tfidf':
            return tfidf.get_tfidf_docs(self, query)
        else:
            print('Wrong input')
        return range(1,3)


#this block deals with the binary weighting. At the moment it returns
#the first 10 documents that have words matching the query
#This block does not weigh the terms as binary gives only present/absent i.e 0/1 weight
#NOTE TO ME: I am still sorting the words in the query, should I do that?
class binary:
    def get_docs(self, query):
        
        #the relevant dictionaries and lists are declared
        binary_final = []
        binary_docs = {}
        
        #The loop is used to check whether a document contains a word or not
        for term in sorted(query, key = query.get, reverse = True):
            if term in self.index:
                for value in self.index[term]:
                    binary_docs[value]=1
        
        #This loop sorts the documents in order of their weight
        for k in sorted(binary_docs, key = binary_docs.get, reverse = True):
            binary_final.append(k)
        
        return binary_final
        
#end of binary block


#This block first sorts the query with the most frequent term in the query first
#Then for each term in the query, the documents are provided weight
#based on the number of times the words occur in them (most occuring documents come first)
class tf:
    def get_tfdocs(self, query):
        
        #the relevant dictionaries and lists are declared
        qd = {}
        tf_final = []
        all_docs = {}
        all_terms = {}
        d_square =0
        q_square =0
        
        #This loop puts all the document ids in a dictionary with an equal weight of zero
        for term in self.index:
            for value in self.index[term]:
                all_docs[value]=0
        
        #This loop weighs each document according to the number of times a term in the query is present in it.
        for term in sorted(query, key = query.get, reverse = True):
            if term in self.index:
                for value in self.index[term]:
                    tf_document = float(self.index[term].get(value))
                    tf_query = query.get(term)
                    all_docs[value] += tf_document
                    all_terms[term] = tf_query
                    if value not in qd:
                        qd[value]=tf_document*tf_query
                    else:
                        qd[value]+=tf_document*tf_query
        
        #This computes the total tf of all the terms in the document
        for value in all_docs:
            d_square += all_docs.get(value)
        
        #This computes the total tf of all the terms in the query
        for term in all_terms:
            q_square += all_terms.get(term)
            
        #This computes the similarity score of the document. This is the weight of the document
        for docid in qd:
            qd[docid] = qd.get(docid)/(d_square*q_square)
        
        #This sorts the sictionary of document ids according to their weight into a list
        for k in sorted(qd, key = qd.get, reverse = True):
            tf_final.append(k)
        
        return tf_final
#end of tf block

#This block first sorts the query with the most frequent term in the query first
#Then for each term in the query, the documents are weighted by the formula
#tfidf = tf.idf
#where: tf is number of times a word occurs in the document
#       idf = log(|D|/df)
#       D is the total number of documents in collection
#       df is the number of documents containing the word.
#weight terms highly if
#They are frequent in relevant documents . . . but
#They are infrequent in collection as a whole
class tfidf:
    def get_tfidf_docs(self, query):
        
        #the relevant dictionaries and lists are declared
        tfidf_docs = {}
        tfidf_final = []
        all_docs = {}
        all_terms = {}
        tf_query = 0
        q_square =0
        d_square =0
        
        #This loop puts all the document ids in a dictionary with an equal weight of zero
        for term in self.index:
            for value in self.index[term]:
                all_docs[value]=0
        
        #The total number of documents in the collection is calculated.
        D = len(all_docs)
        
        #This loop weighs each document according to the number of times a term in the query is present in it and
        #the number of times it is present in the whole collection
        for term in query:
            if term in self.index:
                tf_query = query.get(term)
                working_index = self.index[term]
                for value in sorted(working_index, key = working_index.get, reverse = True):
                    tf_document = float(working_index.get(value))
                    df_document = len(working_index)
                    idf_document = math.log(D/df_document)
                    tfidf_document = tf_document*idf_document
                    tfidf_query = tf_query*idf_document
                    all_docs[value] += tfidf_document
                    all_terms[term] = tfidf_query
                    if value not in tfidf_docs:
                        tfidf_docs[value]=tfidf_document*tfidf_query
                    else:
                        tfidf_docs[value]+=tfidf_document*tfidf_query
        
        #This computes the total tfidf of all the terms in the document
        for value in all_docs:
            d_square += all_docs.get(value)
        
        #This computes the total tfidf of all the terms in the query
        for term in all_terms:
            q_square += all_terms.get(term)
        
        #This computes the similarity score of the document. This is the weight of the document
        for docid in tfidf_docs:
            tfidf_docs[docid] = tfidf_docs.get(docid)/(d_square*q_square)
        
        #This sorts the sictionary of document ids according to their weight into a list
        for k in sorted(tfidf_docs, key = tfidf_docs.get, reverse = True):
            tfidf_final.append(k)
        
        return tfidf_final