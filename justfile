generate:
	python main.py && git add output.apkg dictionary words.txt ignored_lemmas.txt && git commit -m "add words" && git push

