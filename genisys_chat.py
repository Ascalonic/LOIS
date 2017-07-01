from __future__ import division

import nltk
import json
import re


class BotIO(object):

	#to make bot say something
	def bot_say (self, response):
		print "Lois : " + response

	#read input from the human
	def bot_listen(self):
		resp = raw_input("You  : ")
		return resp.lower()


class NLProcessor(object):

	#tokenize the input
	def nl_tokenize (self, response):
		return nltk.word_tokenize(response)

	#part-of-speech tag the tokenized input 
	def nl_tag (self, tokenized_resp):
		return nltk.pos_tag(tokenized_resp)

class DictMgr(object):

	#tokenize the input
	def read_dict (self, dict_type):

		data = {}

		with open(dict_type + '.json', 'r') as f:
    			try:
        			data = json.load(f)
    			# if the file is empty the ValueError will be thrown
    			except ValueError:
        			data = {}

		print data

		return data

	#tokenize the input
	def write_dict (self, dict_type, data):
		with open(dict_type + '.json', 'w') as f:
    			json.dump(data, f)

################# Initialise the dictionaries ##################

dict_mgr = DictMgr()
knowledge_db = dict_mgr.read_dict('knowledge')
words_db = dict_mgr.read_dict('words')
context_db = {}

###############################################################


class NLChunker(object):

	#chunk parse a string using an input grammar (regular expr)
	def nl_chunkparse (self, sourcestr, grammar):
		cp = nltk.RegexpParser(grammar)
		result = cp.parse(sourcestr)
		return result


class BotAI(object):

	#check whether test_string contains only nouns (NN.*)
	def not_only_nouns(self, test_string):
		matchObj = False
		for tagged_unit in test_string:
			if "NN" not in tagged_unit[1]:
				return True
		return False

	#get the type of the word - associated with what
	def get_word_type(self, test_string):
		for word_objs in words_db:
			for key in words_db[word_objs]:
				if test_string == key:
					return word_objs
		return 'nonsense'

	#check whether the input has nothing to do with any thing in words database
	def check_for_nonsense(self, test_tok):
		for unit_tok in test_tok:
			if self.get_word_type(unit_tok)!='nonsense':
				return False
		return True

	#modify the word weights based on recent occurance
	def modify_weights(self, word_type, word):
		my_weight = words_db[word_type][word]
		other_weight = 1.0-my_weight
		words_db[word_type][word] += (other_weight / (len(words_db[word_type])-1))
		new_weight = 1.0-words_db[word_type][word]
		for key in words_db[word_type]:
			if key!=word:
				words_db[word_type][key] = (words_db[word_type][key]/other_weight)*new_weight

	#process the input
	def bot_process (self, response):
		nl_processor = NLProcessor()
		resp_tok = nl_processor.nl_tokenize(response) #tokenize
		resp_tagged = nl_processor.nl_tag(resp_tok)   #POS tag

		if self.not_only_nouns(resp_tagged):
		#Not only nouns
			return "Not only nouns"
		else:
		#Only Nouns
			if len(context_db)==0:
				#search words database for what is he/she saying
				if self.check_for_nonsense(resp_tok):
				#things doesn't make any sense
					return "What are you talking about?!"
				else:
				#find the context
					for tok in resp_tok:
						if self.get_word_type(tok)=='greeting':
							self.modify_weights('greeting',tok)
							return "Hello..."

		nl_chunker = NLChunker()
		print(nl_chunker.nl_chunkparse(resp_tagged, "NP: {<DT>?<JJ>*<NN>}"))


bot_io = BotIO()
bot_ai = BotAI()

bot_io.bot_say("Hi, this is Lois. Say \'#quit\' when you want to quit")

said = ""
while(said!="#quit"):
	said = bot_io.bot_listen()
	bot_io.bot_say(bot_ai.bot_process(said))


bot_io.bot_say("Goodbye, It was fun talking to you!")

dict_mgr.write_dict('knowledge', knowledge_db)
dict_mgr.write_dict('words', words_db)
