UNAME	=	$(shell uname)
PDFTEX	=	pdflatex --shell-escape
BIBTEX	=	bibtex
PDFOPN	=	open

DOCS	=	abstract	\
			report.february	\
			report.april	\
			report.may		\
			slides.naps	\
			slides.pfac	\
			slides.april	\
			slides.may

.PHONY: clean

%.pdf: %.tex
	$(PDFTEX) $<
	-$(BIBTEX) $(*F)
	$(PDFTEX) $<
	$(PDFTEX) $<

default:	report.may slides.may

all: $(DOCS)

abstract:	abstract.pdf
	$(PDFOPN) $<

report.february:	report.february.pdf
	$(PDFOPN) $<

report.april:	report.april.pdf
	$(PDFOPN) $<

report.may:	report.may.pdf
	$(PDFOPN) $<

report.may.pdf:	report.may.bib
report.may.pdf:	report.may/01_intro.tex
report.may.pdf:	report.may/03_partitioning.tex
report.may.pdf:	report.may/04_implementation.tex
report.may.pdf:	report.may/06_results.tex
report.may.pdf:	report.may/89_conclusion.tex
report.may.pdf:	report.may/90_environment.tex
report.may.pdf:	report.may/91_methodology.tex
report.may.pdf:	report.may/images/foz_msh.png
report.may.pdf:	report.may/images/foz_p2_msh.png
report.may.pdf:	report.may/images/foz_p4_msh.png

slides.naps:	slides.naps.pdf
	$(PDFOPN) $<

slides.pfac:	slides.pfac.pdf
	$(PDFOPN) $<

slides.april:	slides.april.pdf
	$(PDFOPN) $<

slides.may:	slides.may.pdf
	$(PDFOPN) $<

clean:
	$(RM) *.log *.aux *.toc *.out *.blg *.bbl *.nav *.snm *.synctex.gz
	$(RM) $(DOCS:%=%.pdf)
