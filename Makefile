default: pfac

vim: naps

pfac:
	pdflatex slides.pfac.tex
	open slides.pfac.pdf

naps:
	pdflatex slides.naps.tex
	xdg-open slides.naps.pdf

clean:
	$(RM) *.log *.aux *.toc
