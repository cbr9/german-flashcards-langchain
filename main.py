import hydra
from langchain import OpenAI, PromptTemplate
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    llm = OpenAI(temperature=0.8)

    template = "Your job is to act as a {speaker_language} lexicographer. Define the {target_language} word {word} as accurately as possible, including practical usage information"

    prompt = PromptTemplate(
        template=template,
        input_variables=["target_language", "speaker_language", "word"],
    )

    response = llm(
        prompt.format(
            speaker_language=cfg.speaker_language,
            target_language=cfg.target_language,
            word=cfg.word,
        )
    )

    response = response.strip()
    print(response)


if __name__ == "__main__":
    main()
