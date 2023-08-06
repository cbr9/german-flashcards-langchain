from langchain import PromptTemplate
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from typing import Any
import json
from pathlib import Path
from pydantic import BaseModel
import genanki
import spacy
from typing import Self


class Word(BaseModel):
    word: str
    translations: list[str]
    definition: str
    examples: list[tuple[str, str]]

    @property
    def formatted_examples(self) -> list[str]:
        return [f"{german} - {english}" for german, english in self.examples]


class Deck(genanki.Deck):
    def __init__(self, deck_id=1381290381, name="German Words", description=""):
        super().__init__(deck_id, name, description)

    @property
    def reverse_model(self) -> genanki.Model:
        return genanki.Model(
            model_id=1000000,
            name="Simple + Reverse",
            fields=[
                {"name": "Word"},
                {"name": "Definition"},
                {"name": "Examples"},
            ],
            templates=[
                {
                    "name": "Card 1",
                    "qfmt": "{{Word}}",
                    "afmt": '{{FrontSide}}<hr id="answer">{{Definition}}<br>{{Examples}}',
                },
                {
                    "name": "Card 2",
                    "qfmt": "{{Definition}}",
                    "afmt": '{{FrontSide}}<hr id="answer">{{Word}}<br>{{Examples}}',
                },
            ],
        )

    def __add__(self, word: Word) -> Self:
        reverse_note = genanki.Note(
            model=self.reverse_model,
            fields=[
                word.word,
                word.definition,
                ulify(word.formatted_examples),
            ],
            guid=genanki.guid_for(word.word),
        )

        self.add_note(reverse_note)
        return self


def ulify(elements: list[Any]) -> str:
    string = "<ul>\n"
    string += "\n".join(["<li>" + str(s) + "</li>" for s in elements])
    string += "\n</ul>"
    return string


def main():
    dictionary = Path("dictionary")
    dictionary.mkdir(parents=True, exist_ok=True)

    deck = Deck()
    llm = ChatOpenAI(temperature=0)
    nlp = spacy.load("de_core_news_lg")

    templates = {
        "plural": PromptTemplate(
            input_variables=["word"],
            template="What is the plural form of the German word {word}? Respond only with the plural form.",
        ),
        "verb_inflections": PromptTemplate(
            input_variables=["word"],
            template="What are the present, past, and past participle forms of the German verb {word}? Respond only with each of the forms separated by a comma. For example, for the verb 'machen', you should respond with 'machen, machte, gemacht'. Do not include any auxiliar verb for the participle.",
        ),
        "definition": PromptTemplate(
            input_variables=["word"],
            template="Give a dictionary definition of the German word {word}. The definition should include usage information, such as what it is most usually used for. Respond only with the definition, without mentioning the language, the word or its part-of-speech tag)",
        ),
        "examples": PromptTemplate(
            input_variables=["word"],
            template="Return a JSON object containing one field called examples, whose elements are lists consisting of two elements, the first of which is an example sentence of the German word {word} and the second is its English translation",
        ),
        "translations": PromptTemplate(
            input_variables=["word"],
            template="Give a comma-separated list of accurate translations for the German word {word}",
        ),
    }

    gender2article = {"Masc": "der", "Fem": "die", "Neut": "das"}

    with open(file="words.txt", mode="r", encoding="utf8") as f:
        words = [w.strip() for w in f.readlines()]

    pbar = tqdm(words)
    for word in pbar:
        pbar.set_description(word)
        if not (dictionary / f"{word}.json").exists():
            card_information = {}
            doc = nlp(word)
            token = doc[0]
            pos = token.pos_

            if pos == "VERB":
                card_information["word"] = llm.predict(
                    templates["verb_inflections"].format(word=word)
                )

            elif pos == "NOUN" or pos == "PROPN":
                gender = token.morph.get(field="Gender", default=None)[0]
                article = gender2article[gender]
                plural = llm.predict(templates["plural"].format(word=word))
                card_information[
                    "word"
                ] = f"{article} {word.capitalize()}, die {plural.capitalize()}"
            else:
                card_information["word"] = word

            card_information["definition"] = llm.predict(
                templates["definition"].format(word=word)
            )

            card_information["examples"] = json.loads(
                llm.predict(templates["examples"].format(word=word))
            )["examples"]

            translations = llm.predict(templates["translations"].format(word=word))
            translations = [t.strip() for t in translations.split(",")]
            card_information["translations"] = translations

            card = Word(**card_information)

            with open(
                file=dictionary / f"{word}.json", mode="w", encoding="utf-8"
            ) as f:
                f.write(json.dumps(card_information, indent=4, ensure_ascii=False))
        else:
            with open(file=dictionary / f"{word}.json", mode="r", encoding="utf8") as f:
                card = Word(**json.load(f))

        deck += card

    genanki.Package(deck).write_to_file("output.apkg")


if __name__ == "__main__":
    main()
