from selenium.common.exceptions import NoSuchElementException
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import selenium
import numpy as np


company_url_set=set()

#read csv
f_company_url = open('./company_url_v2.txt','r')
for line in f_company_url:
	company_url_set.add(str(line))
f_company_url.close()


company_url_list = []
browser = webdriver.Chrome(executable_path='./chromedriver')
f_company_url = open('./company_url_v2.txt','a')
url='https://moat.com/advertiser/adobe?utm_source=random-brand'

related_brand_list = []
random_brand_list = None

while True:
	browser.get(url)
	assert 'Moat' in browser.title
	
	xpath_list = [
		"//*[@id='er-app']/div/div[2]/div/div[1]/span[3]/span[2]",
		"//*[@id='er-app']/div/div[2]/div/div[1]/span[3]/span[3]",
		"//*[@id='er-app']/div/div[2]/div/div[1]/span[3]/span[4]",
		"//*[@id='er-app']/div/div[2]/div/div[1]/span[3]/span[5]",
		"//*[@id='er-app']/div/div[2]/div/div[1]/span[3]/span[6]",
	]

	#record how many related orgs doe current org has
	i = 0
	for xpath in xpath_list:
		try:
			#print(xpath)
			related_brand = browser.find_element_by_xpath(xpath)
		except NoSuchElementException:
			break
		i+=1
		#Click related brand
		related_brand.click()
		url=browser.current_url
		company_url_list.append(url)
		print("related brand:")
		if url not in company_url_set:
			print(url)
			company_url_set.add(url)
			f_company_url.write(url+'\n')
			f_company_url.flush()
		time.sleep(1.5)
		browser.back()
		time.sleep(1.5)
		#Clean up list
	related_brand_list=list()
	

	#Click random brand or deep first search: (use random choice to choose)
	selection = np.random.choice(i+2,1)[0]

	#Click random brand
	if selection <= 1:
		random_brand = browser.find_element_by_link_text('Random Brand')
		random_brand.click()
		url=browser.current_url
		company_url_list.append(url)
		print("random brand:")
		#Parse url to non-random suffix
		#example random brand url : https://moat.com/advertiser/flir?utm_source=random-brand
		try:
			url = url.split("?")[0]
			if url not in company_url_set:
				print(url)
				company_url_set.add(url)
				f_company_url.write(url+'\n')
				f_company_url.flush()
		except:
			pass
		time.sleep(1.5)
	else:
		#composite new xpath to dfs
		xpath = "//*[@id='er-app']/div/div[2]/div/div[1]/span[3]/span[" + str(int(selection)) + "]"
		related_brand = browser.find_element_by_xpath(xpath)
		related_brand.click()
		url=browser.current_url
		print("dfs brand:")
		print(url)
		time.sleep(1.5)

browser.quit()