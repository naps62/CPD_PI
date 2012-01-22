include $(ROOTD)/conf/obj.mk

$(LIBD)/$(DIR).a:	$(OBJS)
	@if [ ! -d "$(@D)" ];	\
	then	\
		mkdir "$(@D)";	\
	fi
	$(AR) $(ARFLAGS) $@ $(OBJS)
	
all:	$(LIBD)/$(DIR).a	

clean:
	$(RM) $(LIBD)/$(DIR).a
