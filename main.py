import json
from langchain import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output_parser import BaseOutputParser
from spacy.tokens import Token
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from typing import Any, Iterator, Optional
from pathlib import Path
from pydantic import BaseModel, Field
import genanki
import spacy
from typing import Self
import re

input_variables_pattern = re.compile(pattern=r"\{\w+\}")
gender2article = {"Masc": "der", "Fem": "die", "Neut": "das"}


class Plural(BaseModel):
    plural: str


class Inflection(BaseModel):
    infinitive: str
    past: str
    participle: str


class Example(BaseModel):
    german: str
    english: str


class Examples(BaseModel):
    examples: list[Example]

    def __iter__(self) -> Iterator[Example]:
        return self.examples.__iter__()


def load_template(
    name: str,
    parser: Optional[BaseOutputParser[Any]] = None,
) -> PromptTemplate:
    path = Path("templates") / f"{name}.tmpl"

    with path.open(mode="r", encoding="utf8") as f:
        text = f.read()
        variables = {
            var.replace("{", "").replace("}", "")
            for var in input_variables_pattern.findall(text)
        }

        partial_variables = {}
        if "format_instructions" in variables:
            assert parser is not None
            partial_variables = {
                "format_instructions": parser.get_format_instructions()
            }
            variables.remove("format_instructions")

        return PromptTemplate(
            template=text,
            input_variables=list(variables),
            partial_variables=partial_variables,
        )


class Word(BaseModel):
    word: str
    token: Optional[Token] = Field(exclude=True, default=None)
    definition: str = Field(default="")
    examples: Examples = Field(default_factory=lambda: Examples(examples=[]))

    class Config:
        arbitrary_types_allowed = True

    @property
    def formatted_examples(self) -> list[str]:
        return [
            f"{example.german} - {example.english}"
            for example in self.examples.examples
        ]

    def define(self, llm: BaseLanguageModel) -> Self:
        assert self.token is not None
        template = load_template("definition")
        self.definition = llm.predict(template.format(word=self.word))
        return self

    def get_examples(self, llm: BaseLanguageModel) -> Self:
        assert self.token is not None
        parser = PydanticOutputParser(pydantic_object=Examples)
        template = load_template("examples", parser=parser)
        prediction = llm.predict(template.format(word=self.word))
        self.examples = parser.parse(prediction)
        return self

    def inflect(self, llm: BaseLanguageModel) -> Self:
        assert self.token is not None
        if self.token.pos_ == "VERB":
            parser = PydanticOutputParser(pydantic_object=Inflection)  # type: ignore
            template = load_template(
                "verb_inflections",
                parser=parser,
            )
            inflections: Inflection = parser.parse(
                llm.predict(template.format(word=self.word))
            )
            self.word = f"{inflections.infinitive}, {inflections.past}, {inflections.participle}"
        elif self.token.pos_ in {"NOUN", "PROPN"}:
            template = load_template("plural")
            gender = self.token.morph.get(field="Gender", default=None)[0]
            article = gender2article[gender]
            plural = llm.predict(
                template.format(word=f"article {self.word.capitalize()}")
            )
            self.word = f"{article} {self.word.capitalize()}, die {plural.capitalize()}"

        return self


class Deck(genanki.Deck):
    def __init__(self, deck_id=1381290381, name="German Words", description=""):
        super().__init__(deck_id, name, description)
        self.deck = set()

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
                # {
                #     "name": "Card 2",
                #     "qfmt": "{{Definition}}",
                #     "afmt": '{{FrontSide}}<hr id="answer">{{Word}}<br>{{Examples}}',
                # },
            ],
        )

    def __add__(self, word: Word | None) -> Self:
        if word is not None and word.word not in self.deck:
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
            self.deck.add(word.word)
        return self


def ulify(elements: list[Any]) -> str:
    string = "<ul>\n"
    string += "\n".join(["<li>" + str(s) + "</li>" for s in elements])
    string += "\n</ul>"
    return string


def process_lemma(
    dictionary: Path,
    lemma: str,
    token: Optional[Token] = None,
    llm: Optional[BaseLanguageModel] = None,
) -> Word:
    if not (dictionary / f"{lemma}.json").exists():
        assert token is not None and llm is not None
        word = Word(word=lemma, token=token).define(llm).get_examples(llm).inflect(llm)

        assert word.token is not None

        with open(
            file=dictionary / f"{lemma}.json",
            mode="w",
            encoding="utf-8",
        ) as f:
            f.write(word.model_dump_json(indent=4))
    else:
        with open(file=dictionary / f"{lemma}.json", mode="r", encoding="utf8") as f:
            word = Word(**json.load(f))
    return word


def main():
    dictionary = Path("dictionary")
    dictionary.mkdir(parents=True, exist_ok=True)

    deck = Deck()
    llm = ChatOpenAI(temperature=0.2)
    nlp = spacy.load("de_core_news_lg")

    with open(file="words.txt", mode="r", encoding="utf8") as f:
        lemmas = [w.strip() for w in f.readlines() if not w.startswith("//")]

    pbar = tqdm(lemmas)
    for lemma in pbar:
        pbar.set_description(lemma)
        doc = nlp(lemma)
        try:
            word = process_lemma(dictionary, lemma, doc[0], llm)
            deck += word
        except (KeyboardInterrupt, IndexError):
            continue

        for example in word.examples:
            doc = nlp(example.german)
            for token in doc:
                if (
                    token.pos_ in {"VERB", "NOUN", "ADV", "ADJ"}
                    and token.lemma_.lower() not in word.word
                ):
                    pbar.set_description(token.lemma_)
                    try:
                        deck += process_lemma(dictionary, token.lemma_, token, llm)
                    except (KeyboardInterrupt, IndexError):
                        continue

    genanki.Package(deck).write_to_file("output.apkg")


if __name__ == "__main__":
    main()
