import random
import math
import json
import time
from datetime import datetime, timedelta, date

WORLD_DATE = datetime.now()
WORLD_DAY = WORLD_DATE.day
WORLD_MONTH = WORLD_DATE.month
WORLD_YEAR = WORLD_DATE.year

SCENES = {}
CHARACTERS = {}
HISTORY = { "player": [] }
RELATIONS = { "player": {} }
INVENTORY = { "player": {} }
EQUIPMENT = { "player": { "outfit": "Casual Clothes" } }
STATUSES = { "player": { "health": "10", "health_default": "10" } }
TRAITS = { "player": {} }

AIUI_FUNCS = None
IMG_IS_CHAR = False
IMG_FILE = ''

# RP TOOL FUNCTIONS

def roll_dice(max: int=6, min: int=1) -> int:
	"""
	Returns a random integer from min to max (inclusive).
    
	Args:
		max: The maximum possible result. Optional (default=6)
		min: The minimum possible result. Optional (default=1)
	"""
	return random.randint(min, max)

def show_image(prompt: str, target: str=''):
	"""
	Generates an image from a prompt using AI and then displays it to the player.
	Use to help the player visual characters and events taking place in the roleplay world.
	Will generate an image of the target character if a target is provided.
    
	Args:
		prompt: A brief description of the image
		target: The name of a character. Optional.
	"""
	global IMG_FILE
	if target == '':
		IMG_FILE = AIUI_FUNCS['GenImageIPA'](prompt, None, IMG_IS_CHAR)
	elif target in CHARACTERS:
		target_img = CHARACTERS[target+'_images'][0]
		IMG_FILE = AIUI_FUNCS['GenImageIPA'](prompt, target_img, True)
		CHARACTERS[target+'_images'].append(IMG_FILE)
	else:
		IMG_FILE = "ERROR: target character not found"

def get_date() -> str:
	"""Get the current date of the roleplay world in year-month-day format."""
	return '{year}-{month}-{day}'.format(year=WORLD_YEAR, month=WORLD_MONTH, day=WORLD_DAY)

def set_date(day: int, month: int, year: int):
	"""
	Set the current date of the roleplay world.
    
	Args:
		day: The day of the month (starting at 1)
		month: The month of the year (starting at 1)
		year: The year
	"""
	global WORLD_DAY, WORLD_MONTH, WORLD_YEAR, WORLD_DATE
	WORLD_DAY = day
	WORLD_MONTH = month
	WORLD_YEAR = year
	WORLD_DATE = date(WORLD_YEAR, WORLD_MONTH, WORLD_DAY)

def shift_date(mins: int, hours: int=0, days: int=0) -> str:
	"""
	Shift the current date of the roleplay world by a specified number of days, months, and years.
	Negative values can be used to shift the date backwards in order to move back in time.
	Returns the new date in year-month-day format.
    
	Args:
		mins: The number of minutes the date is shifted by
		hours: The number of hours the date is shifted by. Optional (default=0)
		days: The number of days the date is shifted by. Optional (default=0)
	"""
	global WORLD_DAY, WORLD_MONTH, WORLD_YEAR, WORLD_DATE
	WORLD_DATE += timedelta(minutes=mins, hours=hours, days=days)
	WORLD_DAY = WORLD_DATE.day
	WORLD_MONTH = WORLD_DATE.month
	WORLD_YEAR = WORLD_DATE.year
	return get_date()

def get_stats(target: str='player') -> dict:
	"""
	Returns all the status data of the target.
    
	Args:
		target: The name of a character, place, or object. Optional (default='player')
	"""
	if target in STATUSES:
		return STATUSES[target]
	return {}

def get_stat(stat_name: str, target: str='player') -> str:
	"""
	Returns the status value of the target.
	Use to get information about the status/state of people, places, and objects.
    
	Args:
		stat_name: The status name
		target: The name of a character, place, or object. Optional (default='player')
	"""
	if target in STATUSES and stat_name in STATUSES[target]:
		return STATUSES[target][stat_name]
	else:
		return ''

def set_stat(stat_name: str, stat_value: str, target: str='player', set_default: bool=True):
	"""
	Sets the status value of the target.
	Use to save information about the status of people such as their health or mood.
	Can also be used to save information about the status of places and objects.
	If stat_value is an empty string the status will be removed from the target.
    
	Args:
		stat_name: The status name
		stat_value: The status value
		target: The name of a character, place, or object. Optional (default='player')
		set_default: If true the default status value is updated. Optional (default=True)
	"""
	global STATUSES
	if target in STATUSES:
		if stat_value == '':
			if stat_name in STATUSES[target]:
				del STATUSES[target][stat_name]
		else:
			STATUSES[target][stat_name] = stat_value
			if set_default or not stat_name+'_default' in STATUSES[target]:
				STATUSES[target][stat_name+'_default'] = stat_value
	else:
		STATUSES[target] = { stat_name: stat_value, stat_name+'_default': stat_value }

def reset_stat(stat_name: str, target: str='player'):
	"""
	Resets the status value of the target back to the default value.
    
	Args:
		stat_name: The status name
		target: The name of a character, place, or object. Optional (default='player')
	"""
	global STATUSES
	if target in STATUSES and stat_name in STATUSES[target]:
		STATUSES[target][stat_name] = STATUSES[target][stat_name+'_default']

def shift_stat(stat_name: str, shift_value: float, target: str='player', limit: bool=True) -> str:
	"""
	Shifts the status value of the target and returns the new value.
	The status value must be numeric for this function to work.
    
	Args:
		stat_name: The status name
		shift_value: The amount added to the status value, can be negative
		target: The name of a character, place, or object. Optional (default='player')
		limit: If true the status value is bound between 0 and the default value. Optional (default=True)
	"""
	global STATUSES
	try:
		status_val = float(STATUSES[target][stat_name])+shift_value
		if limit:
			max_val = float(STATUSES[target][stat_name+'_default'])
			if status_val >= 0.0:
				STATUSES[target][stat_name] = str(min(max(0.0, max_val), status_val))
			else:
				STATUSES[target][stat_name] = str(max(min(0.0, max_val), status_val))
		else:
			STATUSES[target][stat_name] = str(status_val)
		return STATUSES[target][stat_name]
	except:
		return 'error'

def get_traits(target: str='player') -> dict:
	"""
	Returns all the traits of the target.
    
	Args:
		target: The name of a character, place, or object. Optional (default='player')
	"""
	if target in TRAITS:
		return TRAITS[target]
	return {}

def get_trait(trait_name: str, target: str='player') -> str:
	"""
	Returns the trait summary of the target.
	Use to get information about the traits of people, places, and objects.
    
	Args:
		trait_name: The trait name
		target: The name of a character, place, or object. Optional (default='player')
	"""
	if target in TRAITS and trait_name in TRAITS[target]:
		return TRAITS[target][trait_name]
	else:
		return ''

def set_trait(trait_name: str, summary: str, target: str='player'):
	"""
	Sets the trait summary of the target.
	Use to save information about the traits of people such as skills and weaknesses.
	Can also be used to give traits to places and objects.
	If summary is an empty string the trait will be removed.
    
	Args:
		trait_name: The trait name
		summary: A brief description of the trait
		target: The name of a character, place, or object. Optional (default='player')
	"""
	global TRAITS
	if target in TRAITS:
		if summary == '':
			if trait_name in TRAITS[target]:
				del TRAITS[target][trait_name]
		else:
			TRAITS[target][trait_name] = summary
	else:
		TRAITS[target] = { trait_name: summary }

def get_items(target: str='player') -> dict:
	"""
	Returns all the items in the target inventory.
    
	Args:
		target: The name of a character or storage item. Optional (default='player')
	"""
	result = {}
	if target in INVENTORY:
		for item_name, item_count in INVENTORY[target].items():
			if item_count > 0: result[item_name] = item_count
	return result

def count_item(item_name: str, target: str='player') -> int:
	"""
	Returns the number of items called item_name in the target inventory.
    
	Args:
		item_name: The name of the item to count
		target: The name of a character or storage item. Optional (default='player')
	"""
	if target in INVENTORY and item_name in INVENTORY[target]:
		return INVENTORY[target][item_name]
	else:
		return 0

def store_item(item_name: str, quantity: int=1, target: str='player', source: str='') -> bool:
	"""
	Stores the specified quantity of items in the target inventory.
	The items will be moved from the source inventory if a source is provided.
	Returns a boolean value indicating whether the items were successfully stored.
    
	Args:
		item_name: The name of the item being stored
		quantity: The number of items to store
		target: The name of a character or storage item which will store the items. Optional (default='player')
		source: The name of a character or storage item where the items are taken from. Optional.
	"""
	global INVENTORY
	if source != '':
		if source in INVENTORY and item_name in INVENTORY[source]:
			if INVENTORY[source][item_name] >= quantity:
				INVENTORY[source][item_name] -= quantity
			else:
				return False
		else:
			return False
		
	if target in INVENTORY:
		if item_name in INVENTORY[target]:
			INVENTORY[target][item_name] += quantity
		else:
			INVENTORY[target][item_name] = quantity
	else:
		INVENTORY[target] = { item_name: quantity }
	
	return True

def use_item(item_name: str, quantity: int=1, target: str='player') -> int:
	"""
	Removes the specified quantity of items from the target inventory.
	Use in scenarios involving consumable items or when discarding items.
	Returns the number of items removed from the inventory.
    
	Args:
		item_name: The name of the item being removed
		quantity: The number of items to remove
		target: The name of a character or storage item. Optional (default='player')
	"""
	global INVENTORY
	if target in INVENTORY and item_name in INVENTORY[target]:
		if INVENTORY[target][item_name] >= quantity:
			INVENTORY[target][item_name] = max(0, INVENTORY[target][item_name] - quantity)
		else:
			quantity = INVENTORY[target][item_name]
			INVENTORY[target][item_name] = 0
	else:
		return 0
	return quantity

def equip_item(item_name: str, item_slot: str='outfit', target: str='player', source: str='') -> bool:
	"""
	Equips an item to an item slot on the target.
	Removes the equipped item from the source inventory if a source is provided.
	Returns a boolean value indicating whether the item was successfully equipped.
    
	Args:
		item_name: The name of the item being equipped
		item_slot: The name of the item slot, e.g. feet. Optional (default='outfit')
		target: The name of the character equipping the item. Optional (default='player')
		source: The name of a character or storage item. Optional.
	"""
	global EQUIPMENT
	if source != '' and use_item(item_name, 1, source) < 1: return False
	if target in EQUIPMENT:
		if item_slot in EQUIPMENT[target]:
			if EQUIPMENT[target][item_slot] == item_name: return True
			if EQUIPMENT[target][item_slot] != None:
				store_item(EQUIPMENT[target][item_slot], 1, target)
		EQUIPMENT[target][item_slot] = item_name
	else:
		EQUIPMENT[target] = { item_slot: item_name }
	return True

def unequip_item(item_slot: str, target: str='player'):
	"""
	Removes the item equipped to the specified item slot on the target.
	The unequipped item will be moved to the target inventory.
    
	Args:
		item_slot: The name of the item slot, e.g. head.
		target: The name of the character equipping the item. Optional (default='player')
	"""
	global EQUIPMENT
	if target in EQUIPMENT:
		if item_slot in EQUIPMENT[target] and EQUIPMENT[target][item_slot] != None:
			store_item(EQUIPMENT[target][item_slot], 1, target)
		EQUIPMENT[target][item_slot] = None

def get_equipment(target: str='player') -> dict:
	"""
	Returns all the items equipped to the target.
    
	Args:
		target: The name of a character. Optional (default='player')
	"""
	if target in EQUIPMENT:
		return EQUIPMENT[target]
	return {}

def get_relations(target: str='player') -> dict:
	"""
	Returns all the relation data of the target.
    
	Args:
		target: The name of a character, place, or object. Optional (default='player')
	"""
	if target in RELATIONS:
		return RELATIONS[target]
	return {}

def get_relation(target: str, source: str='player') -> str:
	"""
	Returns a summary of the relationship between the source and target.
    
	Args:
		target: The name of a character, place, or object
		source: The name of a character, place, or object. Optional (default='player')
	"""
	if source in RELATIONS and target in RELATIONS[source]:
		return RELATIONS[source][target]
	else:
		return ''

def set_relation(summary: str, target: str, source: str='player'):
	"""
	Sets the relationship between the source and target.
	Can be used to create or update a relationship between two people.
	Can also be used to set a relationship between a person and a place, or person and object, etc.
	If summary is an empty string the relationship will be removed.
    
	Args:
		summary: A brief description of the relationship
		target: The name of a character, place, or object
		source: The name of a character, place, or object. Optional (default='player')
	"""
	global RELATIONS
	if source in RELATIONS:
		if summary == '':
			if target in RELATIONS[source]:
				del RELATIONS[source][target]
		else:
			RELATIONS[source][target] = summary
	else:
		RELATIONS[source] = { target: summary }

def get_npcs() -> list:
	"""Returns a list with the names of all characters with a biography."""
	result = []
	for character, bio in CHARACTERS.items():
		if character != 'player' and not character.endswith('_images'):
			result.append(character)
	return result
	
def get_bio(target: str='player') -> str:
	"""
	Returns the biography of the target character.
    
	Args:
		target: The name of a character. Optional (default='player')
	"""
	if target in CHARACTERS:
		return CHARACTERS[target]
	else:
		return ''

def set_bio(summary: str, target: str='player'):
	"""
	Sets the biography of the target character.
	Use to save information about characters such as their personality and gender.
	If the bio doesn't exist already it will call show_image using the summary as the prompt.
    
	Args:
		summary: The character biography
		target: The name of a character. Optional (default='player')
	"""
	global CHARACTERS, IMG_FILE
	CHARACTERS[target] = summary
	if target+"_image" in CHARACTERS:
		IMG_FILE = ''
	else:
		show_image("Portrait shot of "+target+". "+summary)
		CHARACTERS[target+'_images'] = [IMG_FILE]

def get_history(target: str='player', max: int=8) -> list:
	"""
	Returns the history of the target.
    
	Args:
		target: The name of a character, place, or object. Optional (default='player')
		max: The maximum number of entries to return. Optional (default=8)
	"""
	result = []
	if target in HISTORY and len(HISTORY[target]) > 0:
		result = HISTORY[target][-max:]
	return result

def add_history(summary: str, target: str='player'):
	"""
	Adds a new entry to the history of the target.
	Use to save information about the history of a person, place, or object.
    
	Args:
		summary: A brief description of past events
		target: The name of a character, place, or object. Optional (default='player')
	"""
	global HISTORY
	if target in HISTORY:
		HISTORY[target].append(summary)
	else:
		HISTORY[target] = [summary]

def get_places() -> list:
	"""Returns a list with the names of all scene locations."""
	result = []
	for location, summary in SCENES.items():
		if not location.endswith('_images'):
			result.append(location)
	return result

def get_scene(location: str='current') -> str:
	"""
	Returns a summary of the scene at the specified location.
	Returns a summary of the current scene if a location is not provided.
    
	Args:
		location: The name of a place. Optional.
	"""
	if location in SCENES and len(SCENES[location]) > 0:
		return SCENES[location][-1]
	else:
		return ''

def set_scene(summary: str, location: str='current'):
	"""
	Sets the current scene.
	This function will call show_image using the summary as the prompt.
    
	Args:
		summary: A brief description of the scene
		location: The name of the place the scene is set in. Optional.
	"""
	global SCENES
	if location in SCENES:
		SCENES[location].append(summary)
	else:
		SCENES[location] = [summary]
	show_image(summary)
	if location+'_images' in SCENES:
		SCENES[location+'_images'].append(IMG_FILE)
	else:
		SCENES[location+'_images'] = [IMG_FILE]

def get_targets(data_store: str='stats') -> list:
	"""
	Returns a list of all the target names in data_store.
	data_store must be one of the following data storage names:
	stats, traits, items, equipment, relations, history
    
	Args:
		data_store: The name of the data storage. Optional (default='stats')
	"""
	result = []
	if data_store == 'stats':
		for target in STATUSES: result.append(target)
	elif data_store == 'traits':
		for target in TRAITS: result.append(target)
	elif data_store == 'items':
		for target in INVENTORY: result.append(target)
	elif data_store == 'equipment':
		for target in EQUIPMENT: result.append(target)
	elif data_store == 'relations':
		for target in RELATIONS: result.append(target)
	elif data_store == 'history':
		for target in HISTORY: result.append(target)
	return result

# FUNCTIONS FOR AIUI ENGINE

def SaveGame(save_file, messages, sys_prompt):
	save_json = json.dumps({
		'WORLD_DAY': WORLD_DAY,
		'WORLD_MONTH': WORLD_MONTH,
		'WORLD_YEAR': WORLD_YEAR,
		'SCENES': SCENES,
		'CHARACTERS': CHARACTERS,
		'HISTORY': HISTORY,
		'RELATIONS': RELATIONS,
		'INVENTORY': INVENTORY,
		'EQUIPMENT': EQUIPMENT,
		'STATUSES': STATUSES,
		'TRAITS': TRAITS,
		'messages': messages,
		'sys_prompt': sys_prompt
	})
	f = open(save_file, "wb")
	f.write(save_json.encode('utf-8'))
	f.close()

def LoadGame(save_file):
	global WORLD_DATE, WORLD_DAY, WORLD_MONTH, WORLD_YEAR, SCENES, CHARACTERS, HISTORY, RELATIONS, INVENTORY, EQUIPMENT, STATUSES, TRAITS
	f = open(save_file, "rb")
	save_data = json.loads(f.read().decode('utf-8'))
	f.close()
	SCENES = save_data['SCENES']
	CHARACTERS = save_data['CHARACTERS']
	HISTORY = save_data['HISTORY']
	RELATIONS = save_data['RELATIONS']
	INVENTORY = save_data['INVENTORY']
	EQUIPMENT = save_data['EQUIPMENT']
	STATUSES = save_data['STATUSES']
	TRAITS = save_data['TRAITS']
	WORLD_DAY = save_data['WORLD_DAY']
	WORLD_MONTH = save_data['WORLD_MONTH']
	WORLD_YEAR = save_data['WORLD_YEAR']
	WORLD_DATE = date(WORLD_YEAR, WORLD_MONTH, WORLD_DAY)
	return save_data
	
def NewGame():
	global WORLD_DATE, WORLD_DAY, WORLD_MONTH, WORLD_YEAR, SCENES, CHARACTERS, HISTORY, RELATIONS, INVENTORY, EQUIPMENT, STATUSES, TRAITS
	WORLD_DATE = datetime.now()
	WORLD_DAY = WORLD_DATE.day
	WORLD_MONTH = WORLD_DATE.month
	WORLD_YEAR = WORLD_DATE.year
	SCENES = {}
	CHARACTERS = {}
	HISTORY = { "player": [] }
	RELATIONS = { "player": {} }
	INVENTORY = { "player": {} }
	EQUIPMENT = { "player": {} }
	STATUSES = { "player": {} }
	TRAITS = { "player": {} }

def CallToolFunc(func_name, func_args, aiui_funcs):
	global AIUI_FUNCS, IMG_IS_CHAR
	try:
		AIUI_FUNCS = aiui_funcs
		if "stat_value" in func_args:
			if isinstance(func_args['stat_value'], int) or isinstance(func_args['stat_value'], float):
				func_args['stat_value'] = str(func_args['stat_value'])
		if "shift_value" in func_args and isinstance(func_args['shift_value'], str):
			func_args['shift_value'] = float(func_args['shift_value'])
		if "quantity" in func_args and isinstance(func_args['quantity'], str):
			func_args['quantity'] = int(func_args['quantity'])
		if "min" in func_args and isinstance(func_args['min'], str):
			func_args['min'] = int(func_args['min'])
		if "max" in func_args and isinstance(func_args['max'], str):
			func_args['max'] = int(func_args['max'])
		if "target" in func_args and isinstance(func_args['target'], str):
			func_args['target'] = func_args['target'].lower()
		if "source" in func_args and isinstance(func_args['source'], str):
			func_args['source'] = func_args['source'].lower()
		if "location" in func_args and isinstance(func_args['location'], str):
			func_args['location'] = func_args['location'].lower()
		if "stat_name" in func_args and isinstance(func_args['stat_name'], str):
			func_args['stat_name'] = func_args['stat_name'].lower()
		if "trait_name" in func_args and isinstance(func_args['trait_name'], str):
			func_args['trait_name'] = func_args['trait_name'].lower()
		if "item_name" in func_args and isinstance(func_args['item_name'], str):
			func_args['item_name'] = func_args['item_name'].lower()
		if "item_slot" in func_args and isinstance(func_args['item_slot'], str):
			func_args['item_slot'] = func_args['item_slot'].lower()
		if "data_store" in func_args and isinstance(func_args['data_store'], str):
			func_args['data_store'] = func_args['data_store'].lower()
		if func_name == "roll_dice":
			if "max" in func_args and "min" in func_args:
				return roll_dice(func_args['max'], func_args['min'])
			elif "max" in func_args:
				return roll_dice(func_args['max'])
			elif "min" in func_args:
				return roll_dice(min=func_args['min'])
			else:
				return roll_dice()
		elif func_name == "get_date":
			return get_date()
		elif func_name == "set_date":
			return set_date(func_args['day'], func_args['month'], func_args['year'])
		elif func_name == "shift_date":
			if "months" in func_args and "years" in func_args:
				return shift_date(func_args['days'], func_args['months'], func_args['years'])
			elif "months" in func_args:
				return shift_date(func_args['days'], func_args['months'])
			elif "years" in func_args:
				return shift_date(func_args['days'], years=func_args['years'])
			else:
				return shift_date(func_args['days'])
		elif func_name == "get_targets":
			if "data_store" in func_args:
				return get_targets(func_args['data_store'])
			else:
				return get_targets()
		elif func_name == "get_stats":
			if "target" in func_args:
				return get_stats(func_args['target'])
			else:
				return get_stats()
		elif func_name == "get_stat":
			if "target" in func_args:
				return get_stat(func_args['stat_name'], func_args['target'])
			else:
				return get_stat(func_args['stat_name'])
		elif func_name == "set_stat":
			if "target" in func_args and "set_default" in func_args:
				return set_stat(func_args['stat_name'], func_args['stat_value'], func_args['target'], func_args['set_default'])
			elif "target" in func_args:
				return set_stat(func_args['stat_name'], func_args['stat_value'], func_args['target'])
			elif "set_default" in func_args:
				return set_stat(func_args['stat_name'], func_args['stat_value'], set_default=func_args['set_default'])
			else:
				return set_stat(func_args['stat_name'], func_args['stat_value'])
		elif func_name == "reset_stat":
			if "target" in func_args:
				return reset_stat(func_args['stat_name'], func_args['target'])
			else:
				return reset_stat(func_args['stat_name'])
		elif func_name == "shift_stat":
			if "target" in func_args and "limit" in func_args:
				return shift_stat(func_args['stat_name'], func_args['shift_value'], func_args['target'], func_args['limit'])
			elif "target" in func_args:
				return shift_stat(func_args['stat_name'], func_args['shift_value'], func_args['target'])
			elif "limit" in func_args:
				return shift_stat(func_args['stat_name'], func_args['shift_value'], limit=func_args['limit'])
			else:
				return shift_stat(func_args['stat_name'], func_args['shift_value'])
		elif func_name == "get_traits":
			if "target" in func_args:
				return get_traits(func_args['target'])
			else:
				return get_traits()
		elif func_name == "get_trait":
			if "target" in func_args:
				return get_trait(func_args['trait_name'], func_args['target'])
			else:
				return get_trait(func_args['trait_name'])
		elif func_name == "set_trait":
			if "target" in func_args:
				return set_trait(func_args['trait_name'], func_args['summary'], func_args['target'])
			else:
				return set_trait(func_args['trait_name'], func_args['summary'])
		elif func_name == "get_items":
			if "target" in func_args:
				return get_items(func_args['target'])
			else:
				return get_items()
		elif func_name == "count_item":
			if "target" in func_args:
				return count_item(func_args['item_name'], func_args['target'])
			else:
				return count_item()
		elif func_name == "store_item":
			item_src = func_args['source'] if "source" in func_args else ''
			if "quantity" in func_args and "target" in func_args:
				return store_item(func_args['item_name'], func_args['quantity'], func_args['target'], item_src)
			elif "quantity" in func_args:
				return store_item(func_args['item_name'], func_args['quantity'], source=item_src)
			elif "target" in func_args:
				return store_item(func_args['item_name'], target=func_args['target'], source=item_src)
			else:
				return store_item(func_args['item_name'], source=item_src)
		elif func_name == "use_item":
			if "quantity" in func_args and "target" in func_args:
				return use_item(func_args['item_name'], func_args['quantity'], func_args['target'])
			elif "quantity" in func_args:
				return use_item(func_args['item_name'], func_args['quantity'])
			elif "target" in func_args:
				return use_item(func_args['item_name'], target=func_args['target'])
			else:
				return use_item()
		elif func_name == "get_equipment":
			if "target" in func_args:
				return get_equipment(func_args['target'])
			else:
				return get_equipment()
		elif func_name == "equip_item":
			item_src = func_args['source'] if "source" in func_args else ''
			if "item_slot" in func_args and "target" in func_args:
				return equip_item(func_args['item_name'], func_args['item_slot'], func_args['target'], item_src)
			elif "item_slot" in func_args:
				return equip_item(func_args['item_name'], func_args['item_slot'], source=item_src)
			elif "target" in func_args:
				return equip_item(func_args['item_name'], target=func_args['target'], source=item_src)
			else:
				return equip_item(func_args['item_name'], source=item_src)
		elif func_name == "unequip_item":
			if "target" in func_args:
				return unequip_item(func_args['item_slot'], func_args['target'])
			else:
				return unequip_item(func_args['item_slot'])
		elif func_name == "get_relations":
			if "target" in func_args:
				return get_relations(func_args['target'])
			else:
				return get_relations()
		elif func_name == "get_relation":
			if "source" in func_args:
				return get_relation(func_args['target'], func_args['source'])
			else:
				return get_relation(func_args['target'])
		elif func_name == "set_relation":
			if "source" in func_args:
				return set_relation(func_args['summary'], func_args['target'], func_args['source'])
			else:
				return set_relation(func_args['summary'], func_args['target'])
		elif func_name == "get_npcs":
			return get_npcs()
		elif func_name == "get_bio":
			if "target" in func_args:
				return get_bio(func_args['target'])
			else:
				return get_bio()
		elif func_name == "set_bio":
			IMG_IS_CHAR = True
			if "target" in func_args:
				set_bio(func_args['summary'], func_args['target'])
			else:
				set_bio(func_args['summary'])
			return IMG_FILE
		elif func_name == "get_history":
			if "target" in func_args and "max" in func_args:
				return get_history(func_args['target'], func_args['max'])
			elif "target" in func_args:
				return get_history(func_args['target'])
			elif "max" in func_args:
				return get_history(max=func_args['max'])
			else:
				return get_history()
		elif func_name == "add_history":
			if "target" in func_args:
				return add_history(func_args['summary'], func_args['target'])
			else:
				return add_history(func_args['summary'])
		elif func_name == "get_places":
			return get_places()
		elif func_name == "get_scene":
			if "location" in func_args:
				return get_scene(func_args['location'])
			else:
				return get_scene()
		elif func_name == "set_scene":
			IMG_IS_CHAR = False
			if "location" in func_args:
				set_scene(func_args['summary'], func_args['location'])
			else:
				set_scene(func_args['summary'])
			return IMG_FILE
		elif func_name == "show_image":
			IMG_IS_CHAR = False
			if "target" in func_args:
				show_image(func_args['prompt'], func_args['target'])
			else:
				show_image(func_args['prompt'])
			return IMG_FILE
		else:
			return "ERROR: unknown function"
	except:
		return "ERROR: invalid argument"

def GetToolFuncs():
	return [
		get_date, set_date, shift_date, get_targets, get_stats, get_stat, set_stat, reset_stat, shift_stat,
		get_traits, get_trait, set_trait, get_items, count_item, store_item, use_item, get_equipment,
		equip_item, unequip_item, get_relations, get_relation, set_relation, get_npcs, get_bio, set_bio,
		get_history, add_history, get_places, get_scene, set_scene, show_image, roll_dice
	]
