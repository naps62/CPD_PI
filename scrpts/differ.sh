#!/bin/sh

EXEC1="bin/polu.openmp.time data/xml/tiny.param.xml"
EXEC2="bin/polu.clean data/xml/tiny.param.xml"
OUT_1="data/xml/tiny.polution.xml"
OUT_2="polution.xml"

./${EXEC1}
./${EXEC2}
diff ${OUT_1} ${OUT_2}
