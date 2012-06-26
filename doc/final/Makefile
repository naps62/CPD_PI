UNAME	=	$(shell uname)
PDFTEX	=	pdflatex --shell-escape -halt-on-error
PDFBLD	=	$(PDFTEX) $(*F).tex > /dev/null
BIBTEX	=	bibtex
BIBBLD	=	-$(BIBTEX) $(*F) > /dev/null

ifeq ($(UNAME),Darwin)
PDFOPN	=	open -a /Applications/Adobe\ Reader.app
else
PDFOPN	=	xdg-open
endif

DOCS	=	report slides

.PHONY: clean view slides report

ifeq ($(UNAME),Darwin)
default: slides.pdf
else
default: report
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

report: report.pdf
	$(PDFOPN) $<

report.aux: report/100-intro.tex
report.aux: report/200-case.tex
report.aux: report/210-algorithm.tex
report.aux: report/220-parallelism.tex
report.aux: report/300-sequential.tex
report.aux: report/310-original.tex
report.aux: report/311-simplifications.tex
report.aux: report/320-optimized.tex
report.aux: report/321-aos.tex
report.aux: report/322-soa.tex
report.aux: report/330-parallelization.tex
report.aux: report/400-shared.tex
report.aux: report/410-load.tex
report.aux: report/420-limitations.tex
report.aux: report/500-mpi.tex
report.aux: report/510-partitioning.tex
report.aux: report/520-communication.tex
report.aux: report/530-load-balance.tex
report.aux: report/540-results.tex
report.aux: report/600-cuda.tex
report.aux: report/610-load-balance.tex
report.aux: report/620-optimizations.tex
report.aux: report/630-mesh-reorder.tex
report.aux: report/640-results.tex

slides: slides.pdf
	$(PDFOPN) $<

slides.pdf: slides.tex
	$(PDFTEX) $< > /dev/null

slides.pdf: slides/10-case.tex
slides.pdf: slides/20-sequential.tex
slides.pdf: slides/30-shared.tex
slides.pdf: slides/40-distributed.tex
slides.pdf: slides/50-gpu.tex
slides.pdf: slides/60-finalresults.tex
slides.pdf: slides/90-conclusion.tex

all: $(DOCS)

clean:
	$(RM) *.log *.aux *.toc *.out *.blg *.bbl *.nav *.snm *.synctex.gz
	$(RM) $(DOCS:%=%.pdf)

