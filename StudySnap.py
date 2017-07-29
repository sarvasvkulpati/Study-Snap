import cherrypy
import base64
import nltk
from watson_developer_cloud import AlchemyLanguageV1
import json
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery

#export GOOGLE_APPLICATION_CREDENTIALS ="**file path to creds**""
#export API_KEY = "**key**"

class Server(object):
	nltk.download('punkt')
	def index(self):
		#This handles the server request from the app
		imageString = cherrypy.request.body.read()
		text = self.ocr(imageString)
		keywords = self.Alchemy(text)
		sentenceList = nltk.sent_tokenize(text)
		finalResult = []
		for sentence in sentenceList:
			qna = self.qna(sentence, keywords)
			finalResult.append(qna)
		print json.dumps(finalResult)
		return json.dumps(finalResult)
	index.exposed = True
	def ocr(self, imageString):
		#The google vision ocr
		credentials = GoogleCredentials.get_application_default()
		service = discovery.build('vision', 'v1', credentials=credentials)
		content = [{
		"image":{
			"content": imageString,
				},
		"features": [
			{
				"type":"TEXT_DETECTION",
				"maxResults":"1"
			}]
		}]
		request = service.images().annotate(body= {
		'requests' : content
		})
		response = request.execute()
		parsedValue = response['responses'][0]['textAnnotations'][0]['description']
		parsedValue = parsedValue.replace("\n", " ")
		return parsedValue
	def Alchemy(self, text):
		#accesses IBM's alchemy
		apiKey = os.environ.get('API_KEY')
		alchemyContainer = AlchemyLanguageV1(api_key=apiKey)
		alchemyRaw = alchemyContainer.keywords(text)
		alchemyDump = json.dumps(alchemyRaw)
		jsonResponse = json.loads(alchemyDump)
		jsonData = jsonResponse["keywords"]
		keywords = self.keywordifier(jsonData)
		return keywords
	def keywordifier(self, data):
		#Finds the best keyword
		keywords = []
		for item in data:
			relevance = float(item.get("relevance"))
			if relevance > 0.5:
				keyword = item.get("text")
				keywords.append(keyword)
		return keywords
	def qna(self, sentence, keywords):
		#creates a question and answer
		for keyword in keywords:
			if sentence.find(keyword) != -1:
				sentence = sentence.replace(keyword, "_"*len(keyword))
				qna = {'question':'', 'answer':''}
				qna['question'] = sentence
				qna['answer'] = keyword
				return qna
cherrypy.config.update({'server.socket_host': '0.0.0.0'})
cherrypy.quickstart(Server())
