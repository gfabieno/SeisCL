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


SeisCL_CL= \
	initialize.cl \
	residuals.cl \
	savebnd2D.cl \
	savebnd3D.cl \
	savefreqs.cl \
	surface2D.cl \
	surface2D_SH.cl \
	surface3D.cl \
	update_adjs2D.cl \
	update_adjs2D_SH.cl \
	update_adjs3D.cl \
	update_adjv2D.cl \
	update_adjv2D_SH.cl \
	update_adjv3D.cl \
	update_s2D.cl \
	update_s2D_SH.cl \
	update_s3D.cl \
	update_v2D.cl \
	update_v2D_SH.cl \
	update_v3D.cl \
	update_v_CPML.cl \
	vout.cl \
	fill_transfer_buff_s.cl \
	fill_transfer_buff_v.cl

SeisCL_CL_Headers = $(SeisCL_CL:%.cl=%.hcl)

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

SeisCL_MPI_OBJ = $(SeisCL_MPI_SRC:%.c=%.o)

%.o: %.c $(SeisCL_CL_Headers)
	$(CC) $(CFLAGS) -c -o $@ $<
	
%.hcl: %.cl cl2cstring
	./cl2cstring $< $@


SeisCL_MPI:	$(SeisCL_CL_Headers) $(SeisCL_MPI_OBJ) 
	$(CC) $(LFLAGS) $(SeisCL_MPI_OBJ) -o SeisCL_MPI
	
clean:
	find . -name "*.o" -exec rm {} \; 
	find . -name "*.c%" -exec rm {} \;
	find . -name "*.bck" -exec rm {} \;
	find . -name "*.hcl" -exec rm {} \;

headers: clean $(SeisCL_CL_Headers)

cl2cstring:
	$(CC) cl2cstring.c -o cl2cstring

all: clean SeisCL_MPI
