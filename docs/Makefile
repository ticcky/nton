DOC = nton.pdf
DOC_SRC = nton.md

%.pdf: %.md
	pandoc $< -o $@ --number-sections --filter pandoc-citeproc

.PHONY: live
live:
	inrun $(DOC_SRC) "make $(DOC)"