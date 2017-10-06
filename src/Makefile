# Makefile for SeisCL


UNAME_S := $(shell uname -s)


    ifeq ($(UNAME_S),Linux)
        CC=nvcc
	LFLAGS= -lm -lmpi -lhdf5 -L/usr/local/hdf5/lib -L/usr/lib64/ -lz -pipe
	CFLAGS= -g -I/usr/local/hdf5/include
    endif
    ifeq ($(UNAME_S),Darwin)
        CC=mpicc
	LFLAGS= -lm -lmpi -lhdf5 -framework opencl -L/opt/local/lib
	CFLAGS= -Wall -O3 -I/opt/local/include
    endif


SeisCL_MPI_SRC= \
	assign_modeling_case.c \
    automatic_kernels.c \
	butterworth.c \
	calc_grad.c \
    clbuf.c \
    clerrors.c \
    clprogram.c \
	comm.c\
    event_dependency.c \
	Free_OpenCL.c \
    Init_cst.c \
	Init_model.c \
	Init_MPI.c \
	Init_OpenCL.c \
    kiss_fft.c \
    kiss_fftr.c \
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
