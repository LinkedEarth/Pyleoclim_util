all:
	# gfortran
	f2py -c f2py_wwz.f90 -m f2py_wwz --fcompiler=gfortran --opt='-fopenmp -Ofast' --f90flags='-static -static-libgfortran -static-libgcc' -lgomp
	# ifort
	# f2py -c f2py_wwz.f90 -m f2py_wwz --fcompiler=intelem --opt='-qopenmp -O3 -mkl=parallel' -liomp5
	# f2py -c f2py_wwz.f90 -m f2py_wwz --fcompiler=intelem --opt='-qopenmp -O3 -mkl=parallel -xHost -ipo -Wl,-rpath,${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/../compiler/lib/' -liomp5
	# f2py -c f2py_wwz.f90 -m f2py_wwz --fcompiler=intelem --opt='-qopenmp -O3 -mkl=parallel -xHost -ipo' -liomp5

clean:
	rm -rf f2py_wwz.cpython*
