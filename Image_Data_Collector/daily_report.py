import csv
import time
import codecs
#open datafile
#f_input = open("C:\Python34\input.csv", 'r')
f_input = open('input.csv', 'r')
print("Cleanup...It will take a while..")
i = 0
for line in f_input:
	if i == 0 :
		line.replace('\0','')
		i+=1
	else:
		break 

input_reader = csv.reader((line.replace('\0','') for line in f_input), delimiter=",")
print(f_input[0])
print("Cleanup Done!")
#read file header/column name
header = next(input_reader)
#Initialize list to store column name
columns = dict()
for h,index in zip(header,range(len(header))):
 columns[h] = index
#Check if there are 115 columns
assert len(columns) == 115
f_output = open("output.csv","w",newline="")
output_writer = csv.writer(f_output, delimiter=',')
output_writer.writerow(header)
setdate = "31/08/2018"
deadline = time.strptime(setdate, "%d/%m/%Y")
for row in input_reader:
 try:
  a=time.strptime(row[columns["PRACTITIONERAFFILENDDATE"]], "%d/%m/%Y") 
 except:
  continue
 if row[columns["NETWORKSTATUS"]] == 'Par' \
   and (row[columns["PRACTITIONERAFFILSTATUS"]] == "PA" and row[columns["PRACTITIONERAFFILSTATUS"]] in ["IP","IC","IA","ID","IF","IH","IM","IN","IO"]) \
   and (row[columns["PRACTITIONERAFFILENDDATE"]] not in [""]) \
   and (time.strptime(row[columns["PRACTITIONERAFFILENDDATE"]], "%d/%m/%Y") > deadline):
  print("Qualified Line")
  output_writer.writerow(row)
 else:
  print("Detect Junk Line")
print("All Done")