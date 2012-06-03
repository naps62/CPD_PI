default: slides_may

vim: default

report:
	pdflatex report.tex
	open report.pdf

report_may:
	pdflatex --shell-escape report.may.tex
	xdg-open report.may.pdf

report_april:
	pdflatex --shell-escape report.april.tex
	open report.april.pdf

report_naps:
	pdflatex report.tex
	xdg-open report.pdf

pfac:
	pdflatex slides.pfac.tex
	open slides.pfac.pdf

naps:
	pdflatex slides.naps.tex
	xdg-open slides.naps.pdf

slides_march:
	pdflatex --shell-escape slides.march.tex
	xdg-open slides.march.pdf

slides_april:
	pdflatex --shel-escape slides.april.tex
	xdg-open slides.april.pdf

slides_may:
	pdflatex --shel-escape slides.may.tex
	xdg-open slides.may.pdf

clean:
	$(RM) *.log *.aux *.toc *.out *.blg *.bbl
