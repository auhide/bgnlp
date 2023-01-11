import re
import time
from typing import List

import requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


BASE_URL = "https://slovored.com"


# RegEx patterns:
GET_WORD_LINKS = r'(\/search\/pravopisen-rechnik\/[^\"]+)"'
GET_WORDS_SECTION = r'class="translation"[^\>]*>\s+<pre>([^~]+)<\/pre>'
GET_WORDS = r"([А-Яа-я]+)\s{10,100}([А-Яа-я\s0-9\,\.]+?)\n"


def scrape():
    driver = webdriver.Chrome()
    # That's just the initial page, for the word 'a' in bulgarian.
    driver.get("https://slovored.com/search/pravopisen-rechnik/%D0%B0")

    words_data = []

    while True:
        try:
            driver, words = _scrape_page(driver)
            words_data.extend(words)
            print(f"Added words: {len(words)}")
        except TimeoutException:
            print("Timeout occurred! Maybe there are no more words to scrape.")
            break

    # Save the data into a csv
    df = pd.DataFrame.from_dict(data=words_data)
    print(df.head())
    df.to_csv("bg-pos.csv", index=False, sep="\t")

    print("Dataset saved!")


def _scrape_page(driver: webdriver.Chrome):
    links = _get_word_links(driver.page_source)
    words_data = []

    # Looping through the links of all words on the current page.
    for link in links:
        html = requests.get(link).text
        # A string with all forms of the current word.
        words = _get_words(html)
        if len(words):
            words_data.extend(_parse_scraped_words(words))
    
    _click_next(driver)

    time.sleep(1)

    return driver, words_data


def _click_next(driver: webdriver.Chrome):
    next_button = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.ID, "nextWords")))
    next_button.location_once_scrolled_into_view
    next_button.click()


def _get_word_links(html: str):
    links = re.findall(GET_WORD_LINKS, html)
    links = [BASE_URL + link for link in links]
    
    return links

def _get_words(html: str):
    words_section = re.findall(GET_WORDS_SECTION, html)[0]
    words = re.findall(GET_WORDS, words_section)
    return words


def _parse_scraped_words(words: List[tuple]):
    result = []
    # Get the lemma:
    lemma = words[0][0]
    
    for word, word_form in words:
        result.append({
            "word": word,
            "lemma": lemma,
            "form": word_form
        })

    return result


if __name__ == "__main__":
    scrape()