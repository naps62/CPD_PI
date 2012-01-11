include $(ROOTD)/conf/config.mk

OBJS		=	$(shell find . -type f | egrep -o "[[:alnum:]_.:\-]+\.o$$")
