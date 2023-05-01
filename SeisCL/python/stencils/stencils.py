from .fdcoefficient import FDCoefficients
import numpy as np
from SeisCL.python.function.kernel import cudacl

def get_header_stencil(order, nd, local_size=None, with_local_ops=False):

    if nd == 1:
        names = ["z"]
    elif nd == 2:
        names = ["z", "x"]
    elif nd == 3:
        names = ["z", "y", "x"]

    coefs = FDCoefficients(order=order, maxrerror=1)
    header = coefs.header()
    if local_size is not None:
        lheader, local_grid = get_local_memory_loader(order, nd, local_size,
                                                      with_ops=with_local_ops)
        header += lheader
    else:
        local_grid = ""
    for name in names:
        header += get_stencil(order, nd, name, forward=True,
                              use_local=local_size is not None)
        header += get_stencil(order, nd, name, forward=False,
                              use_local=local_size is not None)
    return header, local_grid


def get_local_memory_loader(order, nd, local_size, with_ops=False,
                            platform="opencl", with_for_loop=True):

    if nd == 1:
        names = ["z"]
    elif nd == 2:
        names = ["z", "x"]
    elif nd == 3:
        names = ["z", "y", "x"]

    header = "//Load in local memory with the halo\n"
    postr1 = ", ".join(["g.%s" % name for name in names])
    postr2 = postr1.replace("g.", "g.l")
    header += "#define load_local_in(v, lv) lv(%s)=v(%s)\n" % (postr2, postr1)
    header += "#define mul_local_in(v, lv) lv(%s)*=v(%s)\n" % (postr2, postr1)


    if with_for_loop:
        load_local_halox = """
        #define load_local_halox(v, lv) \\
        do{\\
            for (int i=g.lx-FDOH; i<FDOH; i+=g.nlx-2*FDOH)\\
                lv(g.lz, g.ly, i)=v(g.z, g.y, g.x-g.lx+i);\\
            for (int i=g.lx%(g.nlx-2*FDOH) ; i<FDOH; i+=g.nlx-2*FDOH)\\
                lv(g.lz, g.ly, g.nlx-FDOH+i)=v(g.z, g.y, g.x-g.lx+g.nlx-FDOH+i);\\
            } while(0)\n""".strip() + "\n"
    else:
        load_local_halox = """
        #define load_local_halox(v, lv) \\
        do{\\
            if (g.lx<2*FDOH)\\
                lv(g.lz, g.ly, g.lx-FDOH)=v(g.z, g.y, g.x-FDOH);\\
            if (g.lx+g.nlx-3*FDOH<FDOH)\\
                lv(g.lz, g.ly, g.lx+g.nlx-3*FDOH)=v(g.z, g.y, g.x+g.nlx-3*FDOH);\\
            if (g.lx>(g.nlx-2*FDOH-1))\\
                lv(g.lz, g.ly, g.lx+FDOH)=v(g.z, g.y, g.x+FDOH);\\
            if (g.lx-g.nlx+3*FDOH>(g.nlx-FDOH-1))\\
                lv(g.lz, g.ly, g.lx-g.nlx+3*FDOH)=v(g.z, g.y, g.x-g.nlx+3*FDOH);\\
            } while(0)\n""".strip() + "\n"

    if nd == 2:
        load_local_halox = load_local_halox.replace("g.ly, ", "")
        load_local_halox = load_local_halox.replace("g.y, ", "")
    elif nd == 1:
        load_local_halox = ""
    header += load_local_halox
    if with_ops:
        header += load_local_halox.replace("load", "mul").replace("=", "*=")
        header += load_local_halox.replace("load", "add").replace("=", "+=")
        header += load_local_halox.replace("load", "sub").replace("=", "-=")
        header += load_local_halox.replace("load", "div").replace("=", "/=")

    if with_for_loop:
        load_local_haloy = """
        #define load_local_haloy(v, lv) \\
        do{\\
            for (int i=g.ly-FDOH; i<FDOH; i+=g.nly-2*FDOH)\\
                lv(g.lz, i, g.lx)=v(g.z, g.y-g.ly+i, g.x);\\
            for (int i=g.ly%(g.nly-2*FDOH) ; i<FDOH; i+=g.nly-2*FDOH)\\
                lv(g.lz, g.nly-FDOH+i, g.lx)=v(g.z, g.y-g.ly+g.nly-FDOH+i, g.x);\\
            } while(0)\n""".strip() + "\n"
    else:
        load_local_haloy = """
        #define load_local_haloy(v, lv) \\
        do{\\
            if (g.ly<2*FDOH)\\
                lv(g.lz, g.ly-FDOH, g.lx)=v(g.z, g.y-FDOH, g.x);\\
            if (g.ly+g.nly-3*FDOH<FDOH)\\
                lv(g.lz, g.ly+g.nly-3*FDOH, g.lx)=v(g.z, g.y+g.nly-3*FDOH, g.x);\\
            if (g.ly>(g.nly-2*FDOH-1))\\
                lv(g.lz, g.ly+FDOH, g.lx)=v(g.z, g.y+FDOH, g.x);\\
            if (g.ly-g.nly+3*FDOH>(g.nly-FDOH-1))\\
                lv(g.lz, g.ly-g.nly+3*FDOH, g.lx)=v(g.z, g.y-g.nly+3*FDOH, g.x);\\
        } while(0)\n""".strip() + "\n"
    if nd < 3:
        load_local_haloy = ""
    header += load_local_haloy
    if with_ops:
        header += load_local_haloy.replace("load", "mul").replace("=", "*=")
        header += load_local_haloy.replace("load", "add").replace("=", "+=")
        header += load_local_haloy.replace("load", "sub").replace("=", "-=")
        header += load_local_haloy.replace("load", "div").replace("=", "/=")

    if with_for_loop:
        load_local_haloz = """
        #define load_local_haloz(v, lv) \\
        do{\\   
            for (int i=g.lz-FDOH; i<FDOH; i+=g.nlz-2*FDOH)\\
                lv(i, g.ly, g.lx)=v(g.z-g.lz+i, g.y, g.x);\\
            for (int i=g.lz%(g.nlz-2*FDOH) ; i<FDOH; i+=g.nlz-2*FDOH)\\
                lv(g.nlz-FDOH+i, g.ly, g.lx)=v(g.z-g.lz+g.nlz-FDOH+i, g.y, g.x);\\
        } while(0)\n""".strip() + "\n"
    else:
        load_local_haloz = """
        #define load_local_haloz(v, lv) \\
        do{\\
            if (g.lz<2*FDOH)\\
                lv(g.lz-FDOH, g.ly, g.lx)=v(g.z-FDOH, g.y, g.x);\\
            if (g.lz>(g.nlz-2*FDOH-1))\\
                lv(g.lz+FDOH, g.ly, g.lx)=v(g.z+FDOH, g.y, g.x);\\
        } while(0)\n""".strip() + "\n"

    if nd == 1:
        load_local_haloz = load_local_haloz.replace(", g.ly, g.lx", "")
        load_local_haloz = load_local_haloz.replace(", g.y, g.x", "")
    elif nd == 2:
        load_local_haloz = load_local_haloz.replace("g.ly, ", "")
        load_local_haloz = load_local_haloz.replace("g.y, ", "")
    header += load_local_haloz
    if with_ops:
        header += load_local_haloz.replace("load", "mul").replace("=", "*=")
        header += load_local_haloz.replace("load", "add").replace("=", "+=")
        header += load_local_haloz.replace("load", "sub").replace("=", "-=")
        header += load_local_haloz.replace("load", "div").replace("=", "/=")

    header = header.replace("2*FDOH", "%d" % order)
    header = header.replace("FDOH", "%d" % (order//2))

    lvar = "#define lvar"
    if nd == 3:
        lvar += "(z, y, x) lvar[(x) * %d + (y) * %d + (z)]\n"
        lvar = lvar % ((local_size[0] + order) * (local_size[1] + order),
                       (local_size[2] + order))
    elif nd == 2:
        lvar += "(z, x) lvar[(x) * %d + (z)]\n" % (local_size[0] + order)
    elif nd == 1:
        lvar += "(x) lvar[(x)]\n"
    header += lvar


    header +="#define BARRIER %s\n" % cudacl["BARRIER"][platform]

    local_grid_filler = ""
    for ii, name in enumerate(names):
        local_grid_filler += "        g.l%s = get_local_id(%d) + %d;\n" \
                       % (name, ii, order//2)
    for ii, name in enumerate(names):
        local_grid_filler += "    g.nl%s = get_local_size(%d) + %d;\n" \
                                % (name, ii, order)
    total = np.prod([el+order for el in local_size])
    local_grid_filler += "    %s float lvar[%d];\n" \
                         % (cudacl["LOCID"][platform], total)

    return header, local_grid_filler


def get_stencil(order, nd, dim, forward=True, use_local=True):

    if nd == 1:
        posp = "(g.z+%d)"
        posm = "(g.z-%d)"
    elif nd == 2:
        if dim == "x":
            posp = "(g.z, g.x+%d)"
            posm = "(g.z, g.x-%d)"
        elif dim == "z":
            posp = "(g.z+%d, g.x)"
            posm = "(g.z-%d, g.x)"
        else:
            raise ValueError("dim must be x or z for 2D")
    elif nd == 3:
        if dim == "x":
            posp = "(g.z, g.y, g.x+%d)"
            posm = "(g.z, g.y, g.x-%d)"
        elif dim == "y":
            posp = "(g.z, g.y+%d, g.x)"
            posm = "(g.z, g.y-%d, g.x)"
        elif dim == "z":
            posp = "(g.z+%d, g.y, g.x)"
            posm = "(g.z-%d, g.y, g.x)"
        else:
            raise ValueError("dim must be x, y or z for 3D")
    if use_local:
        posp = posp.replace("g.", "g.l")
        posm = posm.replace("g.", "g.l")

    if forward:
        x = 1
        flag = "p"
    else:
        x = 0
        flag = "m"
    dxp = "#define D%s%s(v) (" % (dim, flag)
    dxp += ("+\\\n" + " "*len(dxp)).join(["HC%d*(v%s-v%s)" % (ii+1,
                                                              posp % (ii+x),
                                                              posm % (ii+1-x))
                                          for ii in range(order//2)])
    dxp += ")\n"

    return dxp

