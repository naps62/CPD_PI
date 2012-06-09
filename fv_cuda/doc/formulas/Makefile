FILE=formulas

TEX=$(FILE).tex
PDF=$(FILE).pdf

CMD=pdflatex -file-line-error -halt-on-error $(TEX)

default: compile view

compile:
	$(CMD) && $(CMD)

view:
	xdg-open $(PDF)
