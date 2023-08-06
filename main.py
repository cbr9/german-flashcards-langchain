from genanki.model import cached_property
import hydra
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from omegaconf import DictConfig, OmegaConf
from typing import Any
import json
from pathlib import Path
from pydantic import BaseModel
import genanki
import spacy


class Card(BaseModel):
    word: str
    translations: list[str]
    definition: str
    examples: list[tuple[str, str]]

    def model(self) -> genanki.Model:
        return genanki.Model(
            model_id=1000000,
            name="Simple",
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

    def to_anki(self) -> genanki.Note:
        formatted_examples = []
        for german, english in self.examples:
            formatted_examples.append(f"{german} - {english}")

        return genanki.Note(
            model=self.model(),
            fields=[
                self.word,
                self.definition,
                ulify(formatted_examples),
            ],
        )


class Config(BaseModel):
    words: list[str]


def ulify(elements: list[Any]):
    string = "<ul>\n"
    string += "\n".join(["<li>" + str(s) + "</li>" for s in elements])
    string += "\n</ul>"
    return string


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    outputs = Path("outputs")
    outputs.mkdir(parents=True, exist_ok=True)

    config = Config(**OmegaConf.to_object(cfg))  # type: ignore
    llm = ChatOpenAI(temperature=0)
    nlp = spacy.load("de_core_news_lg")
    deck = genanki.Deck(deck_id=1381290381, name="German Words")

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
            template="Return a JSON object containing one field called examples, whose elements are lists consisting of two elements, the first of which is an example sentence of the German word {word} and the second is its English translation. Provide at least five examples.",
        ),
        "translations": PromptTemplate(
            input_variables=["word"],
            template="Give a comma-separated list of accurate translations for the German word {word}",
        ),
    }

    gender2article = {"Masc": "der", "Fem": "die", "Neut": "das"}

    for word in config.words:
        card_information = {}
        doc = nlp(word)
        token = doc[0]
        pos = token.pos_
        print(token.morph)

        if pos == "VERB":
            card_information["word"] = llm.predict(
                templates["verb_inflections"].format(word=word)
            )

        elif pos == "NOUN":
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

        card = Card(**card_information)
        print(card)

        with open(file=outputs / f"{word}.json", mode="w", encoding="utf-8") as f:
            f.write(json.dumps(card_information, indent=4, ensure_ascii=False))

        deck.add_note(card.to_anki())

    genanki.Package(deck).write_to_file("output.apkg")


if __name__ == "__main__":
    main()
