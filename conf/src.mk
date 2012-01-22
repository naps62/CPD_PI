include $(ROOTD)/conf/config.mk

SRCS		=	$(shell find . -type f | egrep -o "[[:alnum:]_.:\-]+\.(cpp|cc)$$")
SRCS.cpp	=	$(filter %.cpp,$(SRCS))
SRCS.cc		=	$(filter %.cc,$(SRCS))
OBJS		= 	$(SRCS.cpp:%.cpp=%.o) $(SRCS.cc:%.cc=%.o)

$(OBJD)/$(DIR)/%.o:	%.cpp
	@if [ ! -d "$(@D)" ];	\
	then	\
		mkdir "$(@D)";	\
		if [ "$(TEMPLATE)" ];	\
		then	\
			cp "$(TEMPLATED)/$(TEMPLATE)" "$(@D)/Makefile";	\
		fi;	\
	fi
	$(COMPILE.cpp) $(OUTPUT_OPTION) $<

$(OBJD)/$(DIR)/%.o:	%.cc
	@if [ ! -d "$(@D)" ];	\
	then	\
		mkdir "$(@D)";	\
	fi
	$(COMPILE.cc) $(OUTPUT_OPTION) $<

all:	$(OBJS:%=$(OBJD)/$(DIR)/%)

clean:
	$(RM) $(OBJD)/$(DIR)/*.o
