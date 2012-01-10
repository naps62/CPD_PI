ROOTD	=	../..

include $(ROOTD)/conf/config.mk

LIB		=	fv

include $(ROOTD)/conf/$(LIB).mk

$(ROOTD)/$(LIBD)/lib$(LIB).a:	$(OBJS)
	$(AR) $(ARFLAGS) $@ $(OBJS)
	
all:	$(ROOTD)/$(LIBD)/lib$(LIB).a	

clean:
	$(RM) $(ROOTD)/$(LIBD)/lib$(LIB).a
