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

.PHONY: clean

%.aux: %.tex
	$(PDFBLD)

%.pdf: %.aux %.bbl
	$(PDFBLD)

%.bbl: %.aux
	$(BIBBLD)
	$(PDFBLD)

%: %.pdf
	$(PDFOPN) $<

ifeq ($(UNAME),Darwin)
default: report.pdf
else
default: report
endif

report.aux: report/200-case.tex
report.aux: report/210-algorithm.tex

all: $(DOCS)

clean:
	$(RM) *.log *.aux *.toc *.out *.blg *.bbl *.nav *.snm *.synctex.gz
	$(RM) $(DOCS:%=%.pdf)
