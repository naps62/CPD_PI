UNAME	=	$(shell uname)
PDFTEX	=	pdflatex --shell-escape -halt-on-error
PDFBLD	=	$(PDFTEX) $(*F).tex > /dev/null
BIBTEX	=	bibtex
BIBBLD	=	-$(BIBTEX) $(*F) > /dev/null

ifeq ($(UNAME),Darwin)
PDFOPN	=	open -a /Applications/Preview.app
else
PDFOPN	=	xdg-open
endif

DOCS	=	report

.PHONY: clean view

ifeq ($(UNAME),Darwin)
default: report.pdf
else
default: report view
endif

view:
	$(PDFOPN) $(DOCS).pdf

%.aux: %.tex
	$(PDFBLD)

%.pdf: %.aux %.bbl
	$(PDFBLD)

%.bbl: %.aux
	$(BIBBLD)
	$(PDFBLD)

%: %.pdf
	

report.pdf: report/100-intro.tex
report.pdf: report/200-case.tex


report.aux: report/200-case.tex
report.aux: report/210-algorithm.tex
report.aux: report/220-parallelism.tex
report.aux: report/300-sequential.tex
report.aux: report/310-original.tex
report.aux: report/311-simplifications.tex

all: $(DOCS)

clean:
	$(RM) *.log *.aux *.toc *.out *.blg *.bbl *.nav *.snm *.synctex.gz
	$(RM) $(DOCS:%=%.pdf)
