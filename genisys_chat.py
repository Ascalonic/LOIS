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

	#process the input
	def bot_process (self, response):
		nl_processor = NLProcessor()
		resp_tok = nl_processor.nl_tokenize(response) #tokenize
		resp_tagged = nl_processor.nl_tag(resp_tok)   #POS tag

		if self.not_only_nouns(resp_tagged):
		#Not only nouns
			return "No only nouns"
		else:
		#Only Nouns
			if len(context_db)==0:
				#things doesn't make any sense
				return "What are you talking about?!"

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
