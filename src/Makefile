# Makefile for SeisCL


UNAME_S := $(shell uname -s)


ifeq ($(UNAME_S),Linux)
ifeq ($(shell hash h5cc),)
  CC=mpicc
else
  CC=h5cc
endif
LFLAGS= -lm -lrt -lmpi -lOpenCL -lhdf5 -L/usr/local/hdf5/lib -L/usr/lib64/ -lz -pipe
CFLAGS= -g
CFLAGS += $(shell mpicc -showme:compile)
LDFLAGS += $(shell mpicc -showme:link)
endif
ifeq ($(UNAME_S),Darwin)
    CC=mpicc
LFLAGS= -lm -lmpi -lhdf5 -L/opt/local/lib
CFLAGS= -Wall -O3 -I/opt/local/include
endif

# Option to choose cuda or OpenCL
ifeq ($(api),cuda)
$(info Building SeisCL with CUDA)
LFLAGS+= -L$(CUDA_PATH)/lib64 -lnvrtc -lcuda -Wl,-rpath,$(CUDA_PATH)/lib64
CFLAGS+= -I$(CUDA_PATH)/include
else
CFLAGS+= -D__SEISCL__
ifeq ($(UNAME_S),Linux)
LFLAGS+= -lOpenCL
endif
ifeq ($(UNAME_S),Darwin)
LFLAGS+= -framework opencl
endif
$(info Building SeisCL with OpenCL)
endif

SeisCL_CL= \
	savebnd2D.cl \
	savebnd3D.cl \
	surface2D.cl \
    	surface2D_adj.cl \
	surface2D_SH.cl \
	surface3D.cl \
	update_adjs2D.cl \
    	update_adjs2D_half2.cl \
	update_adjs2D_SH.cl \
	update_adjs3D.cl \
	update_adjv2D.cl \
    	update_adjv2D_half2.cl \
	update_adjv2D_SH.cl \
	update_adjv3D.cl \
	update_s2D.cl \
	update_s2D_half2.cl \
	update_s2D_SH.cl \
    	update_s2D_acc.cl \
	update_s3D.cl \
    	update_s3D_half2.cl \
	update_v2D.cl \
	update_v2D_half2.cl \
	update_v2D_SH.cl \
    	update_v2D_acc.cl \
	update_v3D.cl \
    	update_v3D_half2.cl

SeisCL_CL_Headers = $(SeisCL_CL:%.cl=%.hcl)

SeisCL_MPI_SRC= \
	    assign_modeling_case.c \
    	automatic_kernels.c \
	    butterworth.c \
	    calc_grad.c \
    	clbuf.c \
    	clerrors.c \
    	clprogram.c \
    	clmodel.c \
	    comm.c\
    	event_dependency.c \
	    Free_OpenCL.c \
    	Init_cst.c \
	    Init_model.c \
	    Init_MPI.c \
	    Init_OpenCL.c \
    	third_party/KISS_FFT/kiss_fft.c \
    	third_party/KISS_FFT/kiss_fftr.c \
        third_party/SOFI3D/CPML_COEFF.c \
        third_party/SOFI3D/holbergcoeff.c \
	    Out_MPI.c \
	    read_hdf5.c\
	    residuals.c \
	    SeisCL_MPI.c \
	    time_stepping.c \
	    writehdf5.c


SeisCL_MPI_OBJ = $(SeisCL_MPI_SRC:%.c=%.o)

%.o: %.c $(SeisCL_CL_Headers)
	$(CC) $(CFLAGS) -c -o $@ $<
	
%.hcl: %.cl cl2cstring
	./cl2cstring $< $@


SeisCL_MPI:	$(SeisCL_CL_Headers) $(SeisCL_MPI_OBJ)
	$(CC) $(SeisCL_MPI_OBJ) $(LFLAGS) -o SeisCL_MPI
	
clean:
	find . -name "*.o" -exec rm {} \; 
	find . -name "*.c%" -exec rm {} \;
	find . -name "*.bck" -exec rm {} \;
	find . -name "*.hcl" -exec rm {} \;

headers: clean $(SeisCL_CL_Headers)

cl2cstring:
	$(CC) cl2cstring.c -o cl2cstring

all: clean SeisCL_MPI
