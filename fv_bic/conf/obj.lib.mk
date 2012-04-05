include $(ROOTD)/conf/obj.mk

$(ROOTD)/$(LIBD)/lib$(LIB).a:	$(OBJS)
	@if [ ! -d "$(@D)" ];	\
	then	\
		mkdir "$(@D)";	\
	fi
	$(AR) $(ARFLAGS) $@ $(OBJS)
	
all:	$(ROOTD)/$(LIBD)/lib$(LIB).a	

clean:
	$(RM) $(ROOTD)/$(LIBD)/lib$(LIB).a
