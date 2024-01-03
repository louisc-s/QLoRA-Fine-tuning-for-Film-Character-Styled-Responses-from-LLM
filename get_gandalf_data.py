
import requests
from bs4 import BeautifulSoup

url = "https://imsdb.com/scripts/Lord-of-the-Rings-The-Two-Towers.html"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Identify the HTML elements containing the dialogue
dialogue_elements = soup.find_all('p')

# Dictionary to store lines by character
character_lines = {}

current_character = None
previous_character = None

for element in dialogue_elements:
    # Check if the element contains bold text (character's name)
    character_name = element.find('b')

    if character_name:
        # Update the current and previous characters
        previous_character = current_character
        current_character = character_name.text.strip()
    elif current_character == 'GANDALF' and previous_character:
        # Extract the dialogue from the current line
        dialogue = element.get_text(strip=True)

        # Add the dialogue to Gandalf's lines and the corresponding character's lines
        if current_character not in character_lines:
            character_lines[current_character] = []
        character_lines[current_character].append(dialogue)

        if previous_character not in character_lines:
            character_lines[previous_character] = []
        character_lines[previous_character].append(dialogue)

# Print or save the lines by character
for character, lines in character_lines.items():
    print(f"{character}:\n")
    for line in lines:
        print(line)
    print("\n---\n")

    