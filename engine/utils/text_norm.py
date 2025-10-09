import re

def NumPartToText(num_str, del_and=True):
	num_txt = ''
	and_txt = "and "
	
	if len(num_str) == 1:
		num_str = '00' + num_str
	elif len(num_str) == 2:
		num_str = '0' + num_str
	
	if num_str == "000" and del_and: return "zero"
	
	if num_str[0] == '0' and del_and:
		and_txt = ''
	elif num_str[0] == '1':
		num_txt += "one hundred "
	elif num_str[0] == '2':
		num_txt += "two hundred "
	elif num_str[0] == '3':
		num_txt += "three hundred "
	elif num_str[0] == '4':
		num_txt += "four hundred "
	elif num_str[0] == '5':
		num_txt += "five hundred "
	elif num_str[0] == '6':
		num_txt += "six hundred "
	elif num_str[0] == '7':
		num_txt += "seven hundred "
	elif num_str[0] == '8':
		num_txt += "eight hundred "
	elif num_str[0] == '9':
		num_txt += "nine hundred "
	
	if num_str[1] == '0':
		if num_str[2] == '1':
			num_txt += and_txt + "one "
		elif num_str[2] == '2':
			num_txt += and_txt + "two "
		elif num_str[2] == '3':
			num_txt += and_txt + "three "
		elif num_str[2] == '4':
			num_txt += and_txt + "four "
		elif num_str[2] == '5':
			num_txt += and_txt + "five "
		elif num_str[2] == '6':
			num_txt += and_txt + "six "
		elif num_str[2] == '7':
			num_txt += and_txt + "seven "
		elif num_str[2] == '8':
			num_txt += and_txt + "eight "
		elif num_str[2] == '9':
			num_txt += and_txt + "nine "
	elif num_str[1] == '1':
		if num_str[2] == '0':
			num_txt += and_txt + "ten "
		elif num_str[2] == '1':
			num_txt += and_txt + "eleven "
		elif num_str[2] == '2':
			num_txt += and_txt + "twelve "
		elif num_str[2] == '3':
			num_txt += and_txt + "thirteen "
		elif num_str[2] == '4':
			num_txt += and_txt + "fourteen "
		elif num_str[2] == '5':
			num_txt += and_txt + "fifteen "
		elif num_str[2] == '6':
			num_txt += and_txt + "sixteen "
		elif num_str[2] == '7':
			num_txt += and_txt + "seventeen "
		elif num_str[2] == '8':
			num_txt += and_txt + "eighteen "
		elif num_str[2] == '9':
			num_txt += and_txt + "nineteen "
	else:
		if num_str[1] == '2':
			num_txt += and_txt + "twenty "
		elif num_str[1] == '3':
			num_txt += and_txt + "thirty "
		elif num_str[1] == '4':
			num_txt += and_txt + "forty "
		elif num_str[1] == '5':
			num_txt += and_txt + "fifty "
		elif num_str[1] == '6':
			num_txt += and_txt + "sixty "
		elif num_str[1] == '7':
			num_txt += and_txt + "seventy "
		elif num_str[1] == '8':
			num_txt += and_txt + "eighty "
		elif num_str[1] == '9':
			num_txt += and_txt + "ninety "
	
		if num_str[2] == '1':
			num_txt += "one "
		elif num_str[2] == '2':
			num_txt += "two "
		elif num_str[2] == '3':
			num_txt += "three "
		elif num_str[2] == '4':
			num_txt += "four "
		elif num_str[2] == '5':
			num_txt += "five "
		elif num_str[2] == '6':
			num_txt += "six "
		elif num_str[2] == '7':
			num_txt += "seven "
		elif num_str[2] == '8':
			num_txt += "eight "
		elif num_str[2] == '9':
			num_txt += "nine "
	
	return num_txt

def NumberToText(num_str):
	if len(num_str) > 15: return num_str
	num_len = len(num_str)
	num_txt = ''
	dec_str = ''
	
	if num_str[0] == '.':
		dec_str = "point AI_UI_"+num_str[1:]
	else:
		num_parts = num_str.split('.')
		num_str = num_parts[0]
		dec_str = " point AI_UI_"+num_parts[1] if len(num_parts) > 1 else ''
	
		if num_len > 12:
			part_txt = NumPartToText(num_str[:num_len-12])
			if part_txt != '': num_txt += part_txt + "trillion, "
		if num_len > 9:
			part_txt = NumPartToText(num_str[max(0,num_len-12):num_len-9])
			if part_txt != '': num_txt += part_txt + "billion, "
		if num_len > 6:
			part_txt = NumPartToText(num_str[max(0,num_len-9):num_len-6])
			if part_txt != '': num_txt += part_txt + "million, "
		if num_len > 3:
			part_txt = NumPartToText(num_str[max(0,num_len-6):num_len-3])
			if part_txt != '': num_txt += part_txt + "thousand, "
		
		part_str = num_str[max(0,num_len-3):]
		part_txt = NumPartToText(part_str, len(part_str)<4)
		if part_txt != '': num_txt += part_txt
		
	return num_txt.strip(' ') + dec_str

def ReplaceMatches(txt, matches, fill_txt):
	for match in matches:
		num1_txt = NumberToText(match[1])
		num2_txt = NumberToText(match[3])
		match_str = match[0] + match[1] + match[2] + match[3] + match[4]
		tts_str = match[0] + num1_txt + fill_txt + num2_txt + match[4]
		txt = txt.replace(match_str, tts_str, 1)
	
	return txt

def NormalizeText(txt):
	
	txt = txt.replace('+', ' plus ')
	
	#comma separated dollars
	matches = re.findall(r"(^|\s)\$(\d{1}|\d{2}|\d{3}),(\d{3})(,\d{3})?(,\d{3})?(,\d{3})?($|\s|\.|\)|\?|\!)", txt)

	for match in matches:
		if int(match[1]) > 0:
			match_str = match[0] + '$' + match[1] + ',' + match[2] + match[3] + match[4] + match[5] + match[6]
			tts_str = match[0] + match[1] + match[2] + match[3] + match[4] + match[5] + ' dollars' + match[6]
			txt = txt.replace(match_str, tts_str.replace(',', ''), 1)

	#comma separated number
	matches = re.findall(r"(^|\s)(\d{1}|\d{2}|\d{3}),(\d{3})(,\d{3})?(,\d{3})?(,\d{3})?($|\s|\.|\)|\?|\!)", txt)

	for match in matches:
		if int(match[1]) > 0:
			match_str = match[0] + match[1] + ',' + match[2] + match[3] + match[4] + match[5] + match[6]
			txt = txt.replace(match_str, match_str.replace(',', ''), 1)

	#less than
	matches1 = re.findall(r"(^|\s)(\d*\.?\d+)(\s<\s)(\d*\.?\d+)($|\W)", txt)
	matches2 = re.findall(r"(^|\s)(\d*\.?\d+)(<)(\d*\.?\d+)($|\W)", txt)

	txt = ReplaceMatches(txt, matches1, ' less than ')
	txt = ReplaceMatches(txt, matches2, ' less than ')

	#greater than
	matches1 = re.findall(r"(^|\s)(\d*\.?\d+)(\s>\s)(\d*\.?\d+)($|\W)", txt)
	matches2 = re.findall(r"(^|\s)(\d*\.?\d+)(>)(\d*\.?\d+)($|\W)", txt)

	txt = ReplaceMatches(txt, matches1, ' greater than ')
	txt = ReplaceMatches(txt, matches2, ' greater than ')

	#times
	matches1 = re.findall(r"(^|\s)(\d*\.?\d+)(\s\*\s)(\d*\.?\d+)($|\W)", txt)
	matches2 = re.findall(r"(^|\s)(\d*\.?\d+)(\*)(\d*\.?\d+)($|\W)", txt)

	txt = ReplaceMatches(txt, matches1, ' times ')
	txt = ReplaceMatches(txt, matches2, ' times ')

	#divide
	matches1 = re.findall(r"(^|\s)(\d*\.?\d+)(\s\/\s)(\d*\.?\d+)($|\W)", txt)
	matches2 = re.findall(r"(^|\s)(\d*\.?\d+)(\/)(\d*\.?\d+)($|\W)", txt)

	txt = ReplaceMatches(txt, matches1, ' divided by ')
	txt = ReplaceMatches(txt, matches2, ' divided by ')

	#minus
	matches1 = re.findall(r"(^|\s)(\d*\.?\d+)(\s-\s)(\d*\.?\d+)($|\W)", txt)
	matches2 = re.findall(r"(^|\s)(\d*\.?\d+)(-)(\d*\.?\d+)($|\W)", txt)

	txt = ReplaceMatches(txt, matches1, ' minus ')
	txt = ReplaceMatches(txt, matches2, ' minus ')

	#mod
	matches1 = re.findall(r"(^|\s)(\d*\.?\d+)(\s%\s)(\d*\.?\d+)($|\W)", txt)
	matches2 = re.findall(r"(^|\s)(\d*\.?\d+)(%)(\d*\.?\d+)($|\W)", txt)

	txt = ReplaceMatches(txt, matches1, ' mod ')
	txt = ReplaceMatches(txt, matches2, ' mod ')

	#power
	matches = re.findall(r"(^|\s)(\d*\.?\d+)(\^)(\d*\.?\d+)($|\W)", txt)
	txt = ReplaceMatches(txt, matches, ' to the power of ')

	#minus
	matches = re.findall(r"(^|\s)-(\d*\.?\d+)($|\W)", txt)

	for match in matches:
		match_str = match[0] + '-' + match[1] + match[2]
		tts_str = match[0] + 'minus ' + match[1] + match[2]
		txt = txt.replace(match_str, tts_str, 1)

	#dollars
	matches = re.findall(r"(^|\s)\$(\d*\.?\d+)($|\W)", txt)

	for match in matches:
		num_txt = NumberToText(match[1])
		match_str = match[0] + '$' + match[1] + match[2]
		tts_str = match[0] + num_txt + ' dollars' + match[2]
		if float(match[1]) == 1.0:
			tts_str = match[0] + num_txt + ' dollar' + match[2]
		txt = txt.replace(match_str, tts_str, 1)

	#number
	matches = re.findall(r"(^|\s)(\d*\.?\d+)($|\W)", txt)

	for match in matches:
		num_txt = NumberToText(match[1])
		match_str = match[0] + match[1] + match[2]
		tts_str = match[0] + num_txt + match[2]
		txt = txt.replace(match_str, tts_str, 1)
	
	#some repeating chars
	matches = re.findall(r"(~+|@+|#+|\$+|&+|_+|=+)", txt)
	
	for match in matches:
		if len(match) > 2:
			txt = txt.replace(match, match[0], 1)

	return txt.replace(' AI_UI_', ' ').replace('0', ' zero ').replace('1', ' one ').replace('2', ' two ').replace('3', ' three ').replace('4', ' four ').replace('5', ' five ').\
	replace('6', ' six ').replace('7', ' seven ').replace('8', ' eight ').replace('9', ' nine ').replace('(', ' left parentheses ').replace(')', ' right parentheses ').\
	replace('<', ' left angle bracket ').replace('>', ' right angle bracket ').replace('[', ' left square bracket ').replace(']', ' right square bracket ').\
	replace('{', ' left curly bracket ').replace('}', ' right curly bracket ').replace('~', ' tilde ').replace('@', ' at ').replace('#', ' hash ').replace('$', ' dollar ').\
	replace('%', ' percent ').replace('&', ' and ').replace('_', ' underscore ').replace('==', ' equal to ').replace('=', ' equals ').replace('/', ' slash ').\
	replace('\\', ' forward slash ').replace("©", ' copyright ').replace("®", ' registered trademark ').replace('\n', ' ').replace('  ', ' ').strip()
