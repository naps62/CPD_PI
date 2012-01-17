include $(ROOTD)/conf/config.mk

SRCS		=	$(shell find . -type f | egrep -o "[[:alnum:]_.:\-]+(\.cpp|\.cc|\.cu)$$")
SRCS.cpp	=	$(filter %.cpp,$(SRCS))
SRCS.cc		=	$(filter %.cc,$(SRCS))
SRCS.cu		=	$(filter %.cu,$(SRCS))
OBJS		= 	$(SRCS.cpp:%.cpp=%.o) $(SRCS.cc:%.cc=%.o) $(SRCS.cu:%.cu=%.o)
