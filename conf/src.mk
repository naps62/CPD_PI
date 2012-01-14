include $(ROOTD)/conf/config.mk

SRCS		=	$(shell find . -type f | egrep -o "[[:alnum:]_.:\-]+\.(cpp|cc)$$")
SRCS.cpp	=	$(filter %.cpp,$(SRCS))
SRCS.cc		=	$(filter %.cc,$(SRCS))
OBJS		= 	$(SRCS.cpp:%.cpp=%.o) $(SRCS.cc:%.cc=%.o)
