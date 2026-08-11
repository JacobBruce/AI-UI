import re

_ONES = {
	"0": "", "1": "one", "2": "two", "3": "three", "4": "four",
	"5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}
_ONES_DIGIT = {
	"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
	"5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}
_TEENS = {
	"0": "ten", "1": "eleven", "2": "twelve", "3": "thirteen", "4": "fourteen",
	"5": "fifteen", "6": "sixteen", "7": "seventeen", "8": "eighteen", "9": "nineteen",
}
_TENS = {
	"2": "twenty", "3": "thirty", "4": "forty", "5": "fifty",
	"6": "sixty", "7": "seventy", "8": "eighty", "9": "ninety",
}
_HUNDREDS = {
	"1": "one hundred", "2": "two hundred", "3": "three hundred",
	"4": "four hundred", "5": "five hundred", "6": "six hundred",
	"7": "seven hundred", "8": "eight hundred", "9": "nine hundred",
}

# Standalone years spoken in year-style (nineteen ninety, not one thousand nine hundred ninety).
_YEAR_RE = re.compile(r"(?<!\d)(1[7-9]\d{2}|20\d{2}|2100)(?!\d)")
_DATE_YMD_RE = re.compile(
	r"(?<!\d)(\d{4})([-/])(\d{1,2})\2(\d{1,2})(?!\d)"
)
_DATE_DMY_RE = re.compile(
	r"(?<!\d)(\d{1,2})([-/])(\d{1,2})\2(\d{4})(?!\d)"
)
_MONEY_RE = re.compile(
	r"(^|\s)\$((?:\d{1,3}(?:,\d{3})+)|\d+)(\.\d+)?($|\W)"
)
_COMMA_NUM_RE = re.compile(
	r"(^|\s)((?:\d{1,3}(?:,\d{3})+))(\.\d+)?($|\W)"
)


def NumPartToText(num_str, del_and=True):
	"""Convert a 1-3 digit group to words. All-zero groups return '' (not 'zero')."""
	if len(num_str) == 1:
		num_str = "00" + num_str
	elif len(num_str) == 2:
		num_str = "0" + num_str

	# Zero groups in 1,000 / 1,000,000 must be silent - only the whole number is "zero".
	if num_str == "000":
		return ""

	parts = []
	h, t, o = num_str[0], num_str[1], num_str[2]
	and_txt = "" if (h == "0" and del_and) else "and "

	if h in _HUNDREDS:
		parts.append(_HUNDREDS[h])

	if t == "0":
		if o != "0":
			parts.append(and_txt + _ONES[o])
	elif t == "1":
		parts.append(and_txt + _TEENS[o])
	else:
		tens = _TENS.get(t, "")
		if o == "0":
			parts.append(and_txt + tens)
		else:
			parts.append(and_txt + tens + " " + _ONES[o])

	return (" ".join(p for p in parts if p) + " ") if parts else ""


def _DigitsToWords(digits):
	"""Spell each digit (for decimal fractional parts / leftover IDs)."""
	return " ".join(_ONES_DIGIT[d] for d in digits if d in _ONES_DIGIT)


def _TwoDigitToText(n):
	"""0-99 as words (no trailing space)."""
	n = int(n)
	if n < 0 or n > 99:
		return str(n)
	if n == 0:
		return "zero"
	s = f"{n:02d}"
	t, o = s[0], s[1]
	if t == "0":
		return _ONES[o]
	if t == "1":
		return _TEENS[o]
	if o == "0":
		return _TENS[t]
	return _TENS[t] + " " + _ONES[o]


def YearToText(year):
	"""Speak 1700-2100 in year style (1990 -> nineteen ninety)."""
	y = int(year)
	if y < 1700 or y > 2100:
		return NumberToText(str(y))
	if y == 2000:
		return "two thousand"
	if 2001 <= y <= 2009:
		return "two thousand " + _ONES[str(y % 10)]
	if y == 2100:
		return "twenty one hundred"
	if 2010 <= y <= 2099:
		return "twenty " + _TwoDigitToText(y % 100)

	# 1700-1999: nineteen ninety / nineteen hundred / nineteen oh five
	century = y // 100
	rest = y % 100
	left = _TwoDigitToText(century)
	if rest == 0:
		return left + " hundred"
	if rest < 10:
		return left + " oh " + _ONES[str(rest)]
	return left + " " + _TwoDigitToText(rest)


def NumberToText(num_str):
	if len(num_str) > 15:
		return num_str
	num_txt = ""
	dec_str = ""

	if num_str[0] == ".":
		dec_digits = num_str[1:]
		dec_str = (" point " + _DigitsToWords(dec_digits)) if dec_digits else ""
		return ("zero" + dec_str).strip()

	num_parts = num_str.split(".")
	num_str = num_parts[0]
	if len(num_parts) > 1 and num_parts[1] != "":
		dec_str = " point " + _DigitsToWords(num_parts[1])

	# All-zero integer (0, 000, ...)
	if num_str == "" or set(num_str) <= {"0"}:
		return ("zero" + dec_str).strip()

	num_len = len(num_str)
	if num_len > 12:
		part_txt = NumPartToText(num_str[: num_len - 12])
		if part_txt != "":
			num_txt += part_txt + "trillion, "
	if num_len > 9:
		part_txt = NumPartToText(num_str[max(0, num_len - 12) : num_len - 9])
		if part_txt != "":
			num_txt += part_txt + "billion, "
	if num_len > 6:
		part_txt = NumPartToText(num_str[max(0, num_len - 9) : num_len - 6])
		if part_txt != "":
			num_txt += part_txt + "million, "
	if num_len > 3:
		part_txt = NumPartToText(num_str[max(0, num_len - 6) : num_len - 3])
		if part_txt != "":
			num_txt += part_txt + "thousand, "

	part_str = num_str[max(0, num_len - 3) :]
	part_txt = NumPartToText(part_str, len(part_str) < 4)
	if part_txt != "":
		num_txt += part_txt

	return (num_txt.strip(" ,") + dec_str).strip()


def ReplaceMatches(txt, matches, fill_txt):
	for match in matches:
		num1_txt = NumberToText(match[1])
		num2_txt = NumberToText(match[3])
		match_str = match[0] + match[1] + match[2] + match[3] + match[4]
		tts_str = match[0] + num1_txt + fill_txt + num2_txt + match[4]
		txt = txt.replace(match_str, tts_str, 1)

	return txt


def StripMarkdown(txt):
	"""Reduce common markdown to speakable plain text (labels / alt text)."""
	s = str(txt or "")

	# Fenced code blocks - do not read source aloud
	s = re.sub(r"```[\w+-]*\n.*?```", " code block ", s, flags=re.DOTALL)
	s = re.sub(r"~~~[\w+-]*\n.*?~~~", " code block ", s, flags=re.DOTALL)

	# Images: ![alt](url) -> alt (or nothing if empty)
	s = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", s)

	# Links: [label](url) -> label
	s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)

	# Autolinks / bare angle URLs
	s = re.sub(r"<(https?://[^>]+)>", " ", s)

	# Bold / bold+italic (longest first)
	s = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", s)
	s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
	s = re.sub(r"__(.+?)__", r"\1", s)

	# Strikethrough
	s = re.sub(r"~~(.+?)~~", r"\1", s)

	# Inline code
	s = re.sub(r"`([^`]+)`", r"\1", s)

	# Italic (*...* / _..._) - require non-space edges so we don't eat "a * b"
	s = re.sub(r"(?<!\*)\*(?!\*)([^\s*](?:.*?[^\s*])?)\*(?!\*)", r"\1", s)
	s = re.sub(r"(?<![A-Za-z0-9_])_([^_\s](?:[^_]*[^_\s])?)_(?![A-Za-z0-9_])", r"\1", s)

	# Headings / horizontal rules / blockquote markers (line-oriented)
	s = re.sub(r"(?m)^#{1,6}\s+", "", s)
	s = re.sub(r"(?m)^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$", " ", s)
	s = re.sub(r"(?m)^>\s?", "", s)

	# Unordered / ordered list markers at line start
	s = re.sub(r"(?m)^\s*[-*+]\s+", "", s)
	s = re.sub(r"(?m)^\s*\d{1,3}\.\s+", "", s)

	return s


def _ReplaceDates(txt):
	"""Turn numeric dates into words before slash/hyphen math runs."""

	def ymd(m):
		return (
			YearToText(m.group(1))
			+ " "
			+ NumberToText(m.group(3))
			+ " "
			+ NumberToText(m.group(4))
		)

	def dmy(m):
		y = int(m.group(4))
		year_txt = YearToText(y) if 1700 <= y <= 2100 else NumberToText(m.group(4))
		return NumberToText(m.group(1)) + " " + NumberToText(m.group(3)) + " " + year_txt

	# Year-first first so 1990-3-5 is not parsed as day-month-year.
	txt = _DATE_YMD_RE.sub(ymd, txt)
	txt = _DATE_DMY_RE.sub(dmy, txt)
	return txt


def _ReplaceYears(txt):
	def repl(m):
		return YearToText(m.group(1))

	return _YEAR_RE.sub(repl, txt)


def _ReplaceMoney(txt):
	"""$10,000.00 -> ten thousand point zero zero dollars (commas + decimals)."""

	def repl(m):
		raw = m.group(2).replace(",", "") + (m.group(3) or "")
		num_txt = NumberToText(raw)
		try:
			unit = "dollar" if float(raw) == 1.0 else "dollars"
		except ValueError:
			unit = "dollars"
		return m.group(1) + num_txt + " " + unit + m.group(4)

	return _MONEY_RE.sub(repl, txt)


def _StripCommaNumbers(txt):
	"""10,000.50 -> 10000.50 so later number pass keeps the decimal."""

	def repl(m):
		return m.group(1) + m.group(2).replace(",", "") + (m.group(3) or "") + m.group(4)

	return _COMMA_NUM_RE.sub(repl, txt)


def NormalizeText(txt):
	txt = StripMarkdown(str(txt or ""))

	# Dates/years before slash, hyphen, or plain number expansion.
	txt = _ReplaceDates(txt)
	txt = _ReplaceYears(txt)

	txt = txt.replace("+", " plus ")

	# Money (with or without thousands commas / decimals)
	txt = _ReplaceMoney(txt)

	# Plain comma-grouped numbers - drop commas, keep decimals for NumberToText
	txt = _StripCommaNumbers(txt)

	# less than
	matches1 = re.findall(r"(^|\s)(\d*\.?\d+)(\s<\s)(\d*\.?\d+)($|\W)", txt)
	matches2 = re.findall(r"(^|\s)(\d*\.?\d+)(<)(\d*\.?\d+)($|\W)", txt)

	txt = ReplaceMatches(txt, matches1, " is less than ")
	txt = ReplaceMatches(txt, matches2, " is less than ")

	# greater than
	matches1 = re.findall(r"(^|\s)(\d*\.?\d+)(\s>\s)(\d*\.?\d+)($|\W)", txt)
	matches2 = re.findall(r"(^|\s)(\d*\.?\d+)(>)(\d*\.?\d+)($|\W)", txt)

	txt = ReplaceMatches(txt, matches1, " is greater than ")
	txt = ReplaceMatches(txt, matches2, " is greater than ")

	# times
	matches1 = re.findall(r"(^|\s)(\d*\.?\d+)(\s\*\s)(\d*\.?\d+)($|\W)", txt)
	matches2 = re.findall(r"(^|\s)(\d*\.?\d+)(\*)(\d*\.?\d+)($|\W)", txt)

	txt = ReplaceMatches(txt, matches1, " times ")
	txt = ReplaceMatches(txt, matches2, " times ")

	# divide
	matches1 = re.findall(r"(^|\s)(\d*\.?\d+)(\s\/\s)(\d*\.?\d+)($|\W)", txt)
	matches2 = re.findall(r"(^|\s)(\d*\.?\d+)(\/)(\d*\.?\d+)($|\W)", txt)

	txt = ReplaceMatches(txt, matches1, " divided by ")
	txt = ReplaceMatches(txt, matches2, " divided by ")

	# minus
	matches1 = re.findall(r"(^|\s)(\d*\.?\d+)(\s-\s)(\d*\.?\d+)($|\W)", txt)
	matches2 = re.findall(r"(^|\s)(\d*\.?\d+)(-)(\d*\.?\d+)($|\W)", txt)

	txt = ReplaceMatches(txt, matches1, " minus ")
	txt = ReplaceMatches(txt, matches2, " minus ")

	# mod
	matches1 = re.findall(r"(^|\s)(\d*\.?\d+)(\s%\s)(\d*\.?\d+)($|\W)", txt)
	matches2 = re.findall(r"(^|\s)(\d*\.?\d+)(%)(\d*\.?\d+)($|\W)", txt)

	txt = ReplaceMatches(txt, matches1, " mod ")
	txt = ReplaceMatches(txt, matches2, " mod ")

	# power
	matches = re.findall(r"(^|\s)(\d*\.?\d+)(\^)(\d*\.?\d+)($|\W)", txt)
	txt = ReplaceMatches(txt, matches, " to the power of ")

	# unary minus
	matches = re.findall(r"(^|\s)-(\d*\.?\d+)($|\W)", txt)

	for match in matches:
		match_str = match[0] + "-" + match[1] + match[2]
		tts_str = match[0] + "minus " + match[1] + match[2]
		txt = txt.replace(match_str, tts_str, 1)

	# number
	matches = re.findall(r"(^|\s)(\d*\.?\d+)($|\W)", txt)

	for match in matches:
		num_txt = NumberToText(match[1])
		match_str = match[0] + match[1] + match[2]
		tts_str = match[0] + num_txt + match[2]
		txt = txt.replace(match_str, tts_str, 1)

	# some repeating chars
	matches = re.findall(r"(~+|!+|@+|#+|\$+|&+|_+|-+|=+)", txt)

	for match in matches:
		if len(match) > 2:
			txt = txt.replace(match, match[0], 1)

	# Leftover digits (e.g. long IDs skipped by NumberToText) - spell individually
	txt = re.sub(r"\d", lambda m: " " + _DigitsToWords(m.group(0)) + " ", txt)

	txt = (
		txt.replace(" & ", " and ")
		.replace(" && ", " and ")
		.replace(" != ", " is unequal to ")
		.replace(" == ", " is equal to ")
		.replace(" >= ", " is greater than or equal to ")
		.replace(" <= ", " is less than or equal to ")
		.replace(" << ", " left shift ")
		.replace(" >> ", " right shift ")
		.replace("(", " opening parentheses ")
		.replace(")", " closing parentheses ")
		.replace("<", " opening angle bracket ")
		.replace(">", " closing angle bracket ")
		.replace("[", " opening square bracket ")
		.replace("]", " closing square bracket ")
		.replace("{", " opening curly bracket ")
		.replace("}", " closing curly bracket ")
		.replace("~", " tilde ")
		.replace("@", " at ")
		.replace("#", " hash ")
		.replace("$", " dollar ")
		.replace("%", " percent ")
		.replace("&", " ampersand ")
		.replace("_", " underscore ")
		.replace("=", " equals ")
		.replace("/", " slash ")
		.replace("\\", " back slash ")
		.replace("©", " copyright ")
		.replace("®", " registered trademark ")
		.replace("\n", " ")
	)

	return re.sub(r"\s+", " ", txt).strip()
