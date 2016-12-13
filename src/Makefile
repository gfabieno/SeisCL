# Makefile for SeisCL


UNAME_S := $(shell uname -s)


    ifeq ($(UNAME_S),Linux)
        CC=mpicc
	LFLAGS= -lm -lmpi -lOpenCL -lhdf5 -L/usr/local/hdf5/lib -L/usr/lib64/ -lz -pipe
	CFLAGS= -g -I/usr/local/hdf5/include
    endif
    ifeq ($(UNAME_S),Darwin)
        CC=mpicc
	LFLAGS= -lm -lmpi -lhdf5 -framework opencl -L/opt/local/lib
	CFLAGS= -Wall -O3 -I/opt/local/include
    endif



.c.o:
	$(CC) $(CFLAGS) -c $<


SeisCL_MPI_SRC= \
	gpu_util.c \
	read_hdf5.c\
	Init_MPI.c \
	Init_cst.c \
	Init_OpenCL.c \
	gpu_initialize_Fkernel.c \
	CPML_coeff.c\
	Init_model.c \
	time_stepping.c \
	comm.c\
	calc_grad.c \
	Out_MPI.c \
	writehdf5.c \
	Free_MPI.c \
	Free_OpenCL.c \
	SeisCL_MPI.c \
	butterworth.c \
	kiss_fft.c \
	kiss_fftr.c \
	residuals.c \
	Hessian.c \
	Init_surfgrid.c \
	
SeisCL_MPI_OBJ = $(SeisCL_MPI_SRC:%.c=%.o)

SeisCL_MPI:		$(SeisCL_MPI_OBJ)
	$(CC) $(LFLAGS) $(SeisCL_MPI_OBJ) -o SeisCL_MPI
	
clean:
	find . -name "*.o" -exec rm {} \; 
	find . -name "*.c%" -exec rm {} \;
	find . -name "*.bck" -exec rm {} \;

all: clean SeisCL_MPI
